export default function Flourish() {
  return (
    <div className="flourish" aria-hidden="true">
      <span className="flourish-line" />
      <svg width="16" height="16" viewBox="0 0 24 24" className="flourish-flower">
        <circle cx="12" cy="8" r="3.4" />
        <circle cx="15.8" cy="10.76" r="3.4" />
        <circle cx="14.35" cy="15.24" r="3.4" />
        <circle cx="9.65" cy="15.24" r="3.4" />
        <circle cx="8.2" cy="10.76" r="3.4" />
      </svg>
      <span className="flourish-line flourish-line-end" />
    </div>
  );
}
