export default function NasiLemakIcon({ size = 28, className }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      className={className}
      aria-hidden="true"
    >
      <path
        d="M24 7 C27 7 29 10 31 14 C36 22 39 30 38 36 C37 40 33 42 24 42 C15 42 11 40 10 36 C9 30 12 22 17 14 C19 10 21 7 24 7 Z"
        fill="#4f8a5c"
      />
      <path
        d="M24 7 C21 7 19 10 17 14 C12 22 9 30 10 36 C11 40 15 42 24 42 Z"
        fill="#3a6b45"
      />
      <path
        d="M24 9 Q22 25 24 40"
        stroke="#2c5536"
        strokeWidth="1.4"
        fill="none"
        opacity="0.6"
      />
      <path
        d="M9.5 19 Q24 24 38.5 19"
        stroke="#6b4423"
        strokeWidth="2.6"
        fill="none"
        strokeLinecap="round"
      />
      <path
        d="M9 30 Q24 35 39 30"
        stroke="#6b4423"
        strokeWidth="2.6"
        fill="none"
        strokeLinecap="round"
      />
      <ellipse cx="24" cy="11" rx="4.5" ry="3.2" fill="#fdf8ee" />
      <ellipse cx="24" cy="11" rx="2.6" ry="1.8" fill="#f2e9d6" />
    </svg>
  );
}
