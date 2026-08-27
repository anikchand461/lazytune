// Small, dependency-free syntax highlighter tuned for the snippets used in
// the docs page (python signatures/examples + a couple of bash lines).
// Not a general-purpose tokenizer — just enough to read cleanly at a glance.

export type Lang = "python" | "bash" | "signature";

interface Token {
  text: string;
  cls: string;
}

const PY_KEYWORDS =
  /^(from|import|as|def|return|class|if|elif|else|for|while|in|is|not|and|or|with|lambda|yield|pass|raise|try|except|finally)\b/;
const PY_BUILTINS = /^(None|True|False|print|self)\b/;

const COLORS = {
  comment: "text-muted2 italic",
  string: "text-accent2/90",
  number: "text-[#ff9f7a]",
  keyword: "text-accent",
  builtin: "text-accent2",
  func: "text-text",
  param: "text-[#c9a8ff]",
  punct: "text-muted",
  plain: "text-text/85",
  command: "text-accent",
  prompt: "text-muted",
};

function tokenizePython(line: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  const patterns: Array<[RegExp, string]> = [
    [/^#.*/, COLORS.comment],
    [/^"""[\s\S]*?"""/, COLORS.string],
    [/^'''[\s\S]*?'''/, COLORS.string],
    [/^"(?:\\.|[^"\\])*"/, COLORS.string],
    [/^'(?:\\.|[^'\\])*'/, COLORS.string],
    [/^\d+\.?\d*/, COLORS.number],
    [PY_BUILTINS, COLORS.builtin],
    [PY_KEYWORDS, COLORS.keyword],
    [/^[A-Za-z_]\w*(?=\()/, COLORS.func],
    [/^[A-Za-z_]\w*(?==)/, COLORS.param],
    [/^[A-Za-z_]\w*/, COLORS.plain],
    [/^[()[\]{}:,.=]+/, COLORS.punct],
    [/^[+\-*/<>!]+/, COLORS.punct],
    [/^\s+/, COLORS.plain],
  ];

  while (i < line.length) {
    const rest = line.slice(i);
    let matched = false;
    for (const [re, cls] of patterns) {
      const m = rest.match(re);
      if (m && m[0]) {
        tokens.push({ text: m[0], cls });
        i += m[0].length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      tokens.push({ text: rest[0], cls: COLORS.plain });
      i += 1;
    }
  }
  return tokens;
}

function tokenizeBash(line: string): Token[] {
  if (line.trim().startsWith("#")) return [{ text: line, cls: COLORS.comment }];
  const tokens: Token[] = [];
  const m = line.match(/^(\$\s*)?(.*)$/);
  if (!m) return [{ text: line, cls: COLORS.plain }];
  const [, prompt, rest] = m;
  if (prompt) tokens.push({ text: prompt, cls: COLORS.prompt });
  const words = rest.split(/(\s+)/);
  words.forEach((w, idx) => {
    if (idx === 0 && w.trim()) tokens.push({ text: w, cls: COLORS.command });
    else tokens.push({ text: w, cls: COLORS.plain });
  });
  return tokens;
}

export function highlightLine(line: string, lang: Lang): Token[] {
  if (lang === "bash") return tokenizeBash(line);
  return tokenizePython(line);
}
