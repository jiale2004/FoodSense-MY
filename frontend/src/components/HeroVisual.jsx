export default function HeroVisual() {
  return (
    <div className="hero-visual-stage">
      <div className="hero-visual-glow" aria-hidden="true" />
      <div className="hero-visual-frame">
        <span className="corner corner-tl" />
        <span className="corner corner-tr" />
        <span className="corner corner-bl" />
        <span className="corner corner-br" />
        <div className="scan-line" aria-hidden="true" />

        <svg viewBox="0 0 300 300" className="plate-svg" aria-hidden="true">
          <circle cx="150" cy="150" r="142" fill="#fff" />
          <circle cx="150" cy="150" r="142" fill="none" stroke="#ece1d6" strokeWidth="3" />
          <circle cx="150" cy="150" r="114" fill="none" stroke="#f2e9dd" strokeWidth="2" />

          {/* rice mound */}
          <ellipse cx="118" cy="168" rx="72" ry="50" fill="#fdf8ee" stroke="#e9dcc0" strokeWidth="2" />
          <circle cx="95" cy="150" r="2" fill="#e3d4b3" />
          <circle cx="115" cy="140" r="2" fill="#e3d4b3" />
          <circle cx="135" cy="155" r="2" fill="#e3d4b3" />
          <circle cx="105" cy="175" r="2" fill="#e3d4b3" />
          <circle cx="140" cy="180" r="2" fill="#e3d4b3" />

          {/* fried egg */}
          <ellipse cx="208" cy="102" rx="44" ry="35" fill="#fffdf6" stroke="#f0e6d0" strokeWidth="2" />
          <circle cx="208" cy="102" r="17" fill="#f2b229" />
          <circle cx="202" cy="96" r="4" fill="#f8cf6b" />

          {/* sambal */}
          <path
            d="M198 188 Q206 162 232 168 Q254 174 249 198 Q244 220 219 216 Q194 212 198 188 Z"
            fill="#c94f1e"
          />

          {/* cucumber slices */}
          <g transform="rotate(-12 88 222)">
            <ellipse cx="88" cy="222" rx="17" ry="8" fill="#bcd9a8" stroke="#9bc27f" strokeWidth="1.5" />
            <circle cx="88" cy="222" r="3.5" fill="#e7f2dc" />
          </g>
          <g transform="rotate(6 116 230)">
            <ellipse cx="116" cy="230" rx="17" ry="8" fill="#bcd9a8" stroke="#9bc27f" strokeWidth="1.5" />
            <circle cx="116" cy="230" r="3.5" fill="#e7f2dc" />
          </g>

          {/* anchovies & peanuts */}
          <circle cx="238" cy="178" r="6" fill="#a9713a" />
          <circle cx="249" cy="188" r="5" fill="#8a5a2b" />
          <circle cx="231" cy="192" r="5" fill="#a9713a" />
        </svg>

        <div className="bbox bbox-rice" style={{ animationDelay: "0.3s" }}>
          <span className="bbox-label">Nasi Lemak · 96%</span>
        </div>
        <div className="bbox bbox-egg" style={{ animationDelay: "0.55s" }}>
          <span className="bbox-label">Egg · 95%</span>
        </div>
        <div className="bbox bbox-sambal" style={{ animationDelay: "0.8s" }}>
          <span className="bbox-label">Sambal · 92%</span>
        </div>
      </div>
    </div>
  );
}
