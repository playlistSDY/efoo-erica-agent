from datetime import datetime,timezone
from unittest.mock import Mock

from app.indexing import index_pending,knowledge_chunks
from app.settings import Settings
from app.store import Store


def test_catalog_reindex_is_scoped_to_changed_directory(tmp_path):
    settings=Settings(data_dir=tmp_path,database_url="",api_key="test",embedding_dimensions=3,schedule_enabled=False)
    store=Store(settings)
    for scope in ["page:facility-directory","facility:49"]:
        store.replace_knowledge(scope,code="campus",name=scope,url="https://life.hanyang.ac.kr/",
            fetched_at=datetime.now(timezone.utc).isoformat(),payload={},
            chunks=knowledge_chunks(scope,scope,[("목록","공식 자료")],kind="page"))
    client=Mock()
    client.embeddings.create.return_value.data=[Mock(index=0,embedding=[1.,0.,0.])]
    assert index_pending(store,settings,client,scopes=["page:facility-directory"])==1
    assert [row["scope"] for row in store.pending()]==["facility:49"]
    client.reset_mock()
    assert index_pending(store,settings,client,scopes=["page:facility-directory"])==0
    client.embeddings.create.assert_not_called()
