/* GFM from pinned Marked; every rendered fragment passes DOMPurify first. */
(function (global) {
  'use strict';

  const TAGS = ['p','br','strong','em','del','s','blockquote','pre','code',
    'h1','h2','h3','h4','h5','h6','ul','ol','li','a','hr',
    'table','thead','tbody','tfoot','tr','th','td','input'];
  const ATTRIBUTES = ['href','title','class','start','align','type','checked','disabled'];
  let parser;

  // Recognize only display ranges, before GFM can pair their tildes as a
  // strikethrough. Inline tokens leave code blocks/spans and URL destinations
  // untouched; source answers in the conversation DB are never rewritten.
  const WEEKDAY = '[월화수목금토일](?:요일)?';
  const CLOCK = '(?:[01]?\\d|2[0-3]):[0-5]\\d';
  const ISO_DATE = '\\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\\d|3[01])';
  const SEPARATOR = '[ \\t]*~{1,2}[ \\t]*';
  const RANGE_RULES = [
    {kind:'weekday', rule:new RegExp('^' + WEEKDAY + SEPARATOR + WEEKDAY + '(?![가-힣A-Za-z])')},
    {kind:'clock', rule:new RegExp('^' + CLOCK + SEPARATOR + CLOCK + '(?![\\d:])')},
    {kind:'date', rule:new RegExp('^' + ISO_DATE + SEPARATOR + ISO_DATE + '(?!\\d)')},
  ];
  const RANGE_START = new RegExp(RANGE_RULES.map(({rule}) => '(?:' + rule.source.slice(1) + ')').join('|'));

  function validISODate(value) {
    const parsed = new Date(value + 'T00:00:00Z');
    return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
  }

  const displayRange = {
    name:'displayRange', level:'inline',
    start(src) { return src.search(RANGE_START); },
    tokenizer(src, tokens) {
      // Link labels can themselves be literal URLs. Preserve them, as well as
      // their destinations, instead of rewriting any part of a URL.
      if (this.lexer.state.inLink) return;
      const previous = tokens.at(-1)?.raw || '';
      for (const {kind, rule} of RANGE_RULES) {
        const match = rule.exec(src);
        if (!match) continue;
        if (kind === 'weekday' && /[가-힣A-Za-z\d]$/.test(previous)) continue;
        if (kind !== 'weekday' && /[\d:]$/.test(previous)) continue;
        const endpoints = match[0].split(/[ \t]*~{1,2}[ \t]*/);
        if (kind === 'date' && !endpoints.every(validISODate)) continue;
        return {type:'displayRange', raw:match[0], text:endpoints.join('–')};
      }
    },
    // Both endpoints are restricted to Korean weekdays or numeric date/time
    // characters. The complete output still passes DOMPurify below.
    renderer(token) { return token.text; },
  };

  function safeHref(value) {
    if (typeof value !== 'string' || !value.trim()) return null;
    const raw = value.trim();
    // DOMPurify already decodes HTML entities. Also reject percent-encoded or
    // whitespace-obfuscated executable protocols instead of treating them as
    // harmless-looking local paths.
    let decoded = raw;
    for (let count = 0; count < 3; count++) {
      try { const next = decodeURIComponent(decoded); if (next === decoded) break; decoded = next; }
      catch { break; }
    }
    const compact = decoded.replace(/[\s\u0000-\u001f\u007f]/g, '');
    if (/^(?:javascript|vbscript|data|file|blob):/i.test(compact) || /[\u0000-\u001f\u007f]/.test(raw)) return null;
    try {
      const base = /^https?:/.test(global.location?.href || '') ? global.location.href : 'http://localhost/';
      const url = new URL(raw, base);
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return null;
      return url.href;
    } catch { return null; }
  }

  function markdownParser() {
    if (!parser) {
      if (!global.marked?.Marked || !global.marked?.Renderer) throw new Error('Markdown parser unavailable');
      const renderer = new global.marked.Renderer();
      // Photographs come exclusively from the server's cited, locally
      // archived gallery. Model-generated image URLs never trigger requests.
      renderer.image = () => '';
      // Raw HTML is not part of the answer contract. Ignore its tokens before
      // sanitizing the generated Markdown HTML as a second independent gate.
      renderer.html = () => '';
      parser = new global.marked.Marked({gfm:true, breaks:false, pedantic:false, async:false, renderer,
        extensions:[displayRange]});
    }
    return parser;
  }

  function normalizeFragment(fragment) {
    for (const element of fragment.querySelectorAll('*')) {
      if (element.tagName !== 'CODE') element.removeAttribute('class');
      else {
        const language = [...element.classList].find(name => /^language-[a-z0-9_-]{1,32}$/i.test(name));
        if (language) element.className = language;
        else element.removeAttribute('class');
      }
      if (element.hasAttribute('align') && !['left','center','right'].includes(element.getAttribute('align'))) element.removeAttribute('align');
      if (element.tagName === 'A') {
        const href = safeHref(element.getAttribute('href'));
        if (href) {
          element.href = href;
          element.target = '_blank';
          element.rel = 'noopener noreferrer';
        } else element.replaceWith(...element.childNodes);
      }
      if (element.tagName === 'INPUT') {
        element.type = 'checkbox';
        element.disabled = true;
        for (const attribute of [...element.attributes]) {
          if (!['type','disabled','checked'].includes(attribute.name)) element.removeAttribute(attribute.name);
        }
      }
    }
  }

  function addCitations(fragment, sources) {
    const available = new Map();
    for (const source of Array.isArray(sources) ? sources : []) {
      if (!source || typeof source !== 'object') continue;
      const id = String(source.id || '');
      const normalized = /^S\d+$/.test(id) ? id : /^\d+$/.test(id) ? 'S' + id : '';
      const href = safeHref(source.url);
      if (normalized && href) available.set(normalized, {href, title:String(source.title || '공식 근거 자료')});
    }
    if (!available.size) return;
    const walker = document.createTreeWalker(fragment, global.NodeFilter.SHOW_TEXT);
    const texts = [];
    while (walker.nextNode()) texts.push(walker.currentNode);
    for (const text of texts) {
      if (text.parentElement?.closest('a,code,pre')) continue;
      const value = text.nodeValue;
      const matches = [...value.matchAll(/\[(S\d+)\]/g)].filter(match => available.has(match[1]));
      if (!matches.length) continue;
      const replacement = document.createDocumentFragment();
      let position = 0;
      for (const match of matches) {
        replacement.append(document.createTextNode(value.slice(position, match.index)));
        const source = available.get(match[1]);
        const anchor = document.createElement('a');
        anchor.className = 'inline-citation';
        anchor.href = source.href;
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.title = source.title;
        anchor.textContent = match[0];
        replacement.append(anchor);
        position = match.index + match[0].length;
      }
      replacement.append(document.createTextNode(value.slice(position)));
      text.replaceWith(replacement);
    }
  }

  function wrapTables(fragment) {
    for (const table of fragment.querySelectorAll('table')) {
      const wrapper = document.createElement('div');
      wrapper.className = 'markdown-table-scroll';
      wrapper.tabIndex = 0;
      wrapper.setAttribute('role', 'region');
      wrapper.setAttribute('aria-label', '답변 표');
      table.replaceWith(wrapper);
      wrapper.append(table);
    }
  }

  function render(container, text, options={}) {
    if (!container || typeof container.replaceChildren !== 'function') return false;
    const value = typeof text === 'string' ? text : String(text ?? '');
    try {
      if (!global.DOMPurify?.sanitize || global.DOMPurify.isSupported === false) throw new Error('Sanitizer unavailable');
      const html = markdownParser().parse(value.replace(/^[\u200B-\u200F\uFEFF]+/, ''));
      const fragment = global.DOMPurify.sanitize(html, {
        ALLOWED_TAGS:TAGS, ALLOWED_ATTR:ATTRIBUTES,
        ALLOW_DATA_ATTR:false, ALLOW_ARIA_ATTR:false,
        FORBID_TAGS:['img','svg','math','style','script','iframe','object','embed','form','textarea','button'],
        FORBID_ATTR:['style','src','srcset','id','name'],
        RETURN_DOM_FRAGMENT:true, RETURN_TRUSTED_TYPE:false
      });
      normalizeFragment(fragment);
      addCitations(fragment, options.sources);
      wrapTables(fragment);
      container.classList.remove('markdown-fallback');
      container.replaceChildren(fragment);
      return true;
    } catch {
      container.classList.add('markdown-fallback');
      container.replaceChildren(document.createTextNode(value));
      return false;
    }
  }

  global.EfooMarkdown = Object.freeze({render});
})(window);
