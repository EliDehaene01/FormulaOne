// A small in-theme SVG avatar (not an AI-generated raster image — a clean
// vector icon renders crisply at any size, costs nothing to regenerate,
// and matches the app's palette exactly). A headset silhouette reads as
// "race engineer on the radio", fitting the Pit Wall persona.
export function PitWallAvatar({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" className="shrink-0">
      <circle cx="16" cy="16" r="16" fill="#ef4444" />
      <path
        d="M16 8a7 7 0 0 0-7 7v4.5a1.5 1.5 0 0 0 1.5 1.5H12v-6H10.1A5.9 5.9 0 0 1 16 9.6a5.9 5.9 0 0 1 5.9 5.4H20v6h1.5a1.5 1.5 0 0 0 1.5-1.5V15a7 7 0 0 0-7-7Z"
        fill="#fff5f5"
      />
      <rect x="9" y="15" width="3" height="6" rx="1" fill="#fff5f5" />
      <rect x="20" y="15" width="3" height="6" rx="1" fill="#fff5f5" />
      <path d="M16 22c-1.7 0-3-.9-3-2h6c0 1.1-1.3 2-3 2Z" fill="#fff5f5" />
    </svg>
  )
}
