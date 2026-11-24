/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17151C",      // app background: violet-tinted graphite
        panel: "#1E1B25",    // panels
        raised: "#27232F",   // inputs, hover
        line: "#2F2B38",     // hairlines
        text: "#ECE8F2",
        muted: "#9A93A7",
        faint: "#6E677B",
        accent: "#A493FF",   // the "in this room" signal
        "accent-deep": "#7D69F2",
        warn: "#F0B46A",
        danger: "#FF8590",
      },
      fontFamily: { sans: ['"Hanken Grotesk Variable"', "ui-sans-serif", "system-ui", "sans-serif"] },
      fontSize: { "2xs": ["0.6875rem", "1rem"] },
    },
  },
  plugins: [],
};
