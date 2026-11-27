from datetime import date as Date
from collections import deque
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import re
import time
from typing import Literal
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAIError
from pydantic import BaseModel, Field, model_validator

from app.domain import RESTAURANTS
from app.rag import Agent, today
from app.settings import Settings
from app.store import Store
from app.conversations import ConversationStore
from app.date_ranges import DateRange, DateRangeError
from app.refresh import MealRefreshService
from app.scheduler import MealScheduler
from app.indexing import index_pending
from app.notice_refresh import NoticeRefreshService

logger = logging.getLogger("efoo")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=5000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=12)
    date: Date = Field(default_factory=today)
    restaurant: str | None = None
    conversation_id: UUID | None = None
    turn_id: UUID | None = None
    start_date: Date | None = None
    end_date: Date | None = None

    @model_validator(mode="after")
    def complete_range(self):
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("시작일과 종료일을 함께 지정해 주세요.")
        if self.start_date and self.end_date < self.start_date:
            raise ValueError("종료일은 시작일보다 빠를 수 없어요.")
        return self


def create_app(settings=None, store=None, agent=None, *, refresh=None, scheduler=None, notice_refresh=None, agentic=None):
    settings = settings or Settings()
    store = store or Store(settings)
    provided_agent = agent is not None
    agent = agent or Agent(store, settings)
    conversations = ConversationStore(store)
    live_agent = isinstance(agent, Agent)
    refresh = refresh if refresh is not None else (MealRefreshService(store,settings) if live_agent else None)
    scheduler = scheduler if scheduler is not None else (MealScheduler(refresh,settings) if live_agent and refresh else None)
    notice_refresh = notice_refresh if notice_refresh is not None else (NoticeRefreshService(store,settings) if live_agent else None)
    # An injected agent owns its orchestration unless a service/factory is
    # supplied explicitly. Production builds a fresh service per request so
    # its tool callback cannot leak another conversation's context.
    if agentic is None and not provided_agent and live_agent and settings.api_key and settings.agentic_enabled:
        from app.agentic import AgenticService
        agentic = AgenticService

    @asynccontextmanager
    async def lifespan(app):
        if scheduler:
            scheduler.start()
        try:
            yield
        finally:
            if scheduler:
                scheduler.stop()

    app = FastAPI(title="한양 ERICA 학교 안내 RAG 에이전트", version="2.1.0", lifespan=lifespan)
    app.state.refresh, app.state.scheduler = refresh, scheduler
    app.state.notice_refresh = notice_refresh
    app.state.agentic = agentic
    chat_admissions = deque()
    chat_in_flight = 0

    @app.middleware("http")
    async def request_metadata(request, call_next):
        nonlocal chat_in_flight
        request_id, started = uuid4().hex, time.monotonic()
        capped_chat = request.method == "POST" and request.url.path.rstrip("/") == "/api/chat" and settings.chat_hourly_limit
        if capped_chat:
            while chat_admissions and started - chat_admissions[0] >= 3600:
                chat_admissions.popleft()
            if len(chat_admissions) >= settings.chat_hourly_limit:
                retry = max(1, int(3600 - (started - chat_admissions[0])) + 1)
                return JSONResponse({"detail": "실습 서버의 시간당 질문 한도에 도달했어요."},
                                    429, headers={"Retry-After": str(retry), "Cache-Control": "no-store"})
            if chat_in_flight >= 2:
                return JSONResponse({"detail": "답변을 처리 중이에요. 잠시 후 다시 질문해 주세요."},
                                    429, headers={"Retry-After": "5", "Cache-Control": "no-store"})
            chat_admissions.append(started)
            chat_in_flight += 1
        try:
            response = await call_next(request)
        finally:
            if capped_chat:
                chat_in_flight -= 1
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        if request.url.path.startswith("/api/conversations") or request.url.path=="/api/chat":
            response.headers["Cache-Control"]="no-store"
            response.headers["Vary"]="Cookie"
        # Do not log user questions, answers, keys or retrieved document content.
        logger.info("request id=%s method=%s status=%s duration_ms=%.0f",
                    request_id, request.method, response.status_code, (time.monotonic()-started)*1000)
        return response

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    def browser_owner(request:Request,response:Response):
        existing=request.cookies.get("efoo_session")
        token,owner=conversations.ensure_owner(existing)
        if token!=existing:
            response.set_cookie("efoo_session",token,max_age=365*24*60*60,httponly=True,
                secure=settings.secure_session_cookie or request.url.scheme=="https",samesite="lax",path="/")
        return owner

    @app.get("/api/conversations/current")
    def current_conversation(request:Request,response:Response):
        return conversations.current(browser_owner(request,response))

    @app.get("/api/conversations")
    def conversation_list(request:Request,response:Response):
        return {"conversations":conversations.list(browser_owner(request,response))}

    @app.post("/api/conversations")
    def new_conversation(request:Request,response:Response):
        return conversations.create(browser_owner(request,response))

    @app.get("/api/conversations/{conversation_id}")
    def conversation_detail(conversation_id:UUID,request:Request,response:Response):
        try:
            return conversations.get(browser_owner(request,response),str(conversation_id))
        except KeyError:
            raise HTTPException(404,"대화를 찾을 수 없습니다")

    @app.get("/api/status")
    def status():
        stats = store.stats()
        ready = bool(settings.api_key and stats["indexed_count"] and
                     stats["indexed_count"] == stats["chunk_count"])
        return {**stats, "ready": ready, "mode": "openai" if ready else "retrieval",
                "model": settings.model, "embedding_model": settings.embedding_model,
                "agentic_enabled": bool(agentic),
                "collection_policy":{"max_date_range_days":settings.max_date_range_days,
                    "meal_refresh_hours":settings.meal_refresh_hours,
                    "empty_refresh_minutes":settings.meal_empty_refresh_minutes},
                "scheduler":scheduler.status() if scheduler else {"enabled":False,"running":False},
                "today": today().isoformat(),
                "restaurants": [{"code": code, "name": info["name"]} for code, info in RESTAURANTS.items()]}

    @app.get("/api/meals")
    def meals(date: Date = Query(default_factory=today), restaurant: str | None = None):
        if restaurant and restaurant not in RESTAURANTS:
            raise HTTPException(422, "알 수 없는 식당입니다")
        rows = store.collections(date.isoformat(), restaurant)
        by_code = {r["code"]: r for r in rows}
        data = []
        for row in rows:
            for meal in row["meals"]:
                data.append({**meal, "restaurant_code": row["code"], "restaurant_name": row["name"],
                             "source_url": row["url"], "fetched_at": row["fetched_at"]})
        return {"date": date.isoformat(), "meals": data, "collections": [
            {"code": code, "name": info["name"], "status": "collected" if code in by_code else "missing",
             "fetched_at": by_code.get(code, {}).get("fetched_at")}
            for code, info in RESTAURANTS.items() if not restaurant or code == restaurant]}

    @app.get("/api/facilities")
    def facilities():
        return {"facilities":[{"code":r["code"],"name":r["name"],"source_url":r["url"],
            "fetched_at":r["fetched_at"],"info":r["payload"].get("restaurant_info",{})}
            for r in store.knowledge() if r["scope"].startswith("facility:")]}

    @app.get("/media/{filename}")
    def media(filename:str):
        if not re.fullmatch(r"[a-f0-9]{64}\.(jpg|png|gif|webp|avif|pdf|doc|docx|xls|xlsx|ppt|pptx)",filename):
            raise HTTPException(404,"파일을 찾을 수 없습니다")
        path=settings.data_dir/"media"/filename
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404,"파일을 찾을 수 없습니다")
        return FileResponse(path,headers={"Cache-Control":"public, max-age=604800, immutable"})

    def index_directory():
        try:
            index_pending(store,settings,scopes=["page:facility-directory"])
        except Exception:
            logger.warning("Directory indexing incomplete; exact catalog retrieval remains available")

    def with_facility_context(result):
        if "selected_facility_ids" in result:
            return result
        selected=[]
        for record in (result.get("directory") or {}).get("records",[]):
            if record.get("id") is not None and str(record["id"]) not in selected:
                selected.append(str(record["id"]))
        for source in result.get("sources",[]) if not selected else []:
            identity=(source.get("metadata") or {}).get("facility_id")
            if identity is None:
                matched=re.search(r"/(?:shop-)?detail\.php\?id=(\d+)",source.get("url",""))
                identity=matched[1] if matched else None
            if identity is not None and str(identity) not in selected:
                selected.append(str(identity))
        return {**result,"selected_facility_ids":selected,"reference_facility_ids":selected}

    def run_school_query(message,history,selected_date,restaurant,custom_range,background_tasks):
        """One grounded runtime shared by direct questions and agent tools."""
        if not live_agent:
            return agent.chat(message,history,selected_date,restaurant)
        plan=agent.plan(message,history,selected_date,restaurant,date_range=custom_range)
        report=None
        facility_report=(agent.school.refresh_targets(plan.facility_ids)
            if plan.notice_request and plan.facility_ids and not plan.clarification else None)
        school_answer=agent.school.answer(plan.school_plan,refresh=True) if plan.school_plan else None
        directory_answer=agent.directory.answer(plan.directory_query,refresh=True) if plan.directory_query else None
        if directory_answer and settings.api_key:
            background_tasks.add_task(index_directory)
        if refresh and plan.menu_request:
            report=refresh.ensure(plan.span.start,plan.span.end,[plan.code] if plan.code else None)
        notice_report=(notice_refresh.ensure() if notice_refresh and plan.notice_request else None)
        result=agent.chat(message,history,selected_date,restaurant,prepared_plan=plan,refresh_report=report,
                          directory_answer=directory_answer,school_answer=school_answer,notice_report=notice_report,
                          facility_refresh_report=facility_report)
        return with_facility_context(result)

    @app.post("/api/chat")
    def chat(body: ChatRequest,request:Request,response:Response,background_tasks:BackgroundTasks):
        if body.restaurant and body.restaurant not in RESTAURANTS:
            raise HTTPException(422, "알 수 없는 식당입니다")
        if not body.message.strip():
            raise HTTPException(422, "질문을 입력해 주세요")
        try:
            owner=browser_owner(request,response)
            conversation_id=str(body.conversation_id) if body.conversation_id else conversations.current(owner)["id"]
            turn_id=str(body.turn_id or uuid4())
            with conversations.turn_lock(owner,conversation_id):
                conversations.get(owner,conversation_id,mark_current=False)
                try:
                    cached=conversations.get_turn(owner,conversation_id,turn_id,message=body.message)
                except ValueError:
                    raise HTTPException(409,"같은 요청 ID를 다른 질문에 사용할 수 없습니다")
                if cached is not None:
                    if body.start_date and cached.get("request_date_range",cached.get("date_range"))!={"start":body.start_date.isoformat(),"end":body.end_date.isoformat()}:
                        raise HTTPException(409,"같은 요청 ID를 다른 날짜 범위에 사용할 수 없습니다")
                    result=cached
                else:
                    history=conversations.recent_history(owner,conversation_id,reference_date=body.date) if body.conversation_id else [m.model_dump() for m in body.history]
                    try:
                        custom_range=DateRange(body.start_date,body.end_date) if body.start_date else None
                        if custom_range and len(custom_range)>settings.max_date_range_days:
                            raise DateRangeError(f"한 번에 확인할 수 있는 기간은 {settings.max_date_range_days}일까지예요. 기간을 나누어 질문해 주세요.")
                        if agentic:
                            def school_query(query):
                                return run_school_query(query,history,body.date,body.restaurant,custom_range,background_tasks)
                            service=agentic if hasattr(agentic,"run") and not isinstance(agentic,type) else agentic(
                                settings,agent,school_query=school_query)
                            result=with_facility_context(service.run(body.message,history,body.date,body.restaurant,
                                                                     date_range=custom_range))
                        else:
                            result=run_school_query(body.message,history,body.date,body.restaurant,custom_range,background_tasks)
                    except DateRangeError as exc:
                        result={"answer":str(exc),"mode":"clarification","sources":[],"images":[],
                                "model":None,"trace":["날짜 범위 확인 필요"]}
                    except KeyError:
                        raise HTTPException(422,"답변 데이터 형식을 확인할 수 없습니다")
                    if body.start_date:
                        result={**result,"request_date_range":{"start":body.start_date.isoformat(),"end":body.end_date.isoformat()}}
                    conversations.append_turn(owner,conversation_id,body.message,result,turn_id)
                    result=conversations.get_turn(owner,conversation_id,turn_id,message=body.message)
            if body.conversation_id or body.turn_id:
                return {**result,"conversation_id":conversation_id,"turn_id":turn_id}
            return result
        except KeyError:
            raise HTTPException(404,"대화를 찾을 수 없습니다")
        except OpenAIError:
            logger.warning("OpenAI API request failed")
            raise HTTPException(502, "OpenAI 요청에 실패했습니다. 서버의 API 키, 모델 접근 권한과 한도를 확인해 주세요.")
        except (ValueError, TypeError):
            raise HTTPException(422, "날짜 또는 답변 데이터 형식을 확인할 수 없습니다.")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


app = create_app()
