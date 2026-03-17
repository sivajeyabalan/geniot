export default function Panel({ as: Tag = 'section', title, subtitle, children, className = '' }) {
  return (
    <Tag className={`panel ${className}`.trim()}>
      {(title || subtitle) && (
        <header className="panel-header">
          {title ? <h2 className="panel-title">{title}</h2> : null}
          {subtitle ? <p className="panel-subtitle">{subtitle}</p> : null}
        </header>
      )}
      {children}
    </Tag>
  )
}
