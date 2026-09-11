/**
 * Root-cause explanations come back from the LLM as loose markdown
 * (paragraphs + **bold**). Rather than pull in a markdown dependency for
 * one field, render just enough of it: paragraph breaks and bold spans.
 */
export function renderFormattedText(text) {
  if (!text) return null
  return text
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean)
    .map((paragraph, paragraphIndex) => (
      <p key={paragraphIndex}>
        {paragraph.split(/(\*\*[^*]+\*\*)/g).map((chunk, chunkIndex) => {
          const boldMatch = /^\*\*([^*]+)\*\*$/.exec(chunk)
          return boldMatch ? <strong key={chunkIndex}>{boldMatch[1]}</strong> : chunk
        })}
      </p>
    ))
}
