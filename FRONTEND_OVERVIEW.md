# 🎨 MoodCast Frontend Overview

> **Historical design note.** This file describes an early frontend sketch.
> Parts of it no longer match the app: there is no in-page player or
> "Now Playing" bar, and tracks open on Spotify. For current behaviour, see
> [FEATURES.md](FEATURES.md) and [FLASK_SETUP.md](FLASK_SETUP.md).

## 🎭 Your New Beautiful UI is Ready!

### What's Been Created

```
mood_reccomender/
├── flask_app.py              ← Main Flask server (NEW!)
├── start_moodcast.bat        ← Easy startup script (NEW!)
├── templates/                ← HTML templates (NEW!)
│   ├── index.html           ← Home page with mood selection
│   ├── music.html           ← Spotify-style music player
│   └── movies.html          ← Netflix-style movie browser
├── QUICKSTART.md            ← Quick start guide (NEW!)
└── FLASK_SETUP.md           ← Detailed setup guide (NEW!)
```

### Design Features

#### 🏠 Home Page (index.html)
```
┌─────────────────────────────────────────┐
│  MoodCast                               │
│                                         │
│     How are you feeling today?          │
│                                         │
│  😊  😢  🤩  😌  😰                    │
│ Happy Sad Excited Relaxed Anxious       │
│                                         │
│  🎯 Match My Mood  |  ⬆️ Uplift Me     │
│                                         │
│    🎬 Movies    |    🎵 Music          │
└─────────────────────────────────────────┘
```

Features:
- Animated gradient background with floating orbs
- Interactive mood cards with hover effects
- Smooth transitions
- Session storage for mood persistence

#### 🎵 Music Page (music.html)
```
┌─────────────────────────────────────────┐
│ 🎵 Sidebar  │  Main Content             │
│             │                           │
│ MoodTunes   │  🎵 Your Vibe            │
│             │  ━━━━━━━━━━━━━━━━━━━━    │
│ 🏠 Home     │                           │
│ 🎭 Moods    │  ▶ Play                  │
│ 🎬 Movies   │                           │
│             │  Track List:              │
│ Playlists:  │  1. Song Name - Artist   │
│ 😊 Happy    │  2. Song Name - Artist   │
│ 😢 Sad      │  3. Song Name - Artist   │
│ 🤩 Excited  │                           │
│             │                           │
├─────────────┴───────────────────────────┤
│ 🎵 Now Playing: Track - Artist    ▶ ⏸  │
└─────────────────────────────────────────┘
```

Features:
- Spotify-inspired dark theme
- Sidebar with mood playlists
- Interactive player controls
- Progress bar and volume control
- Track list with hover effects

#### 🎬 Movies Page (movies.html)
```
┌─────────────────────────────────────────┐
│  MoodCinema              🎵 Music | Home │
│                                         │
│  ╔═══════════════════════════════════╗  │
│  ║                                   ║  │
│  ║  Featured Movie                   ║  │
│  ║  ★ 8.5 • 2023 • 2h 15m           ║  │
│  ║  Description...                   ║  │
│  ╚═══════════════════════════════════╝  │
│                                         │
│  😊 Happy  😢 Sad  🤩 Excited  😌 Chill │
│                                         │
│  Recommended For You                    │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐   │
│  │🎭  │ │🎪  │ │🌌  │ │🎨  │ │🏛️  │   │
│  │    │ │    │ │    │ │    │ │    │   │
│  └────┘ └────┘ └────┘ └────┘ └────┘   │
└─────────────────────────────────────────┘
```

Features:
- Netflix-inspired dark theme
- Hero banner with featured movie
- Grid layout with movie cards
- Hover effects with details
- Modal for full movie information
- Mood switcher strip

### Color Scheme

**Home Page:**
- Background: Gradient (warm orange to cool blue)
- Accents: Red (#e63946) and Green (#1db954)
- Text: Dark (#080810) on light backgrounds

**Music Page:**
- Background: Spotify Dark (#121212)
- Accent: Spotify Green (#1db954)
- Surface: Dark gray (#181818)

**Movies Page:**
- Background: Netflix Dark (#141414)
- Accent: Netflix Red (#e50914)
- Surface: Darker gray (#1f1f1f)

### Animations

1. **Floating Orbs** - Smooth drift animation on home page
2. **Particle Effects** - Floating particles in background
3. **Card Hover** - Scale and shadow effects
4. **Fade In** - Staggered content appearance
5. **Pulse** - Subtle glow effects

### How to Start

**Easiest Way:**
```bash
# Just double-click this file:
start_moodcast.bat
```

**Command Line:**
```bash
python flask_app.py
```

**Then open:**
```
http://localhost:5000
```

### Integration with Your Code

The Flask app (`flask_app.py`) integrates seamlessly with your existing code:

```python
# Uses your existing functions:
from src.data_loader import load_music_catalog, load_movie_data
from src.recommend import recommend_music, recommend_movies

# Loads your data:
music_df = load_music_catalog()   # the 278k catalog
movie_df = load_movie_data()

# Gets recommendations:
recs = recommend_music(music_df, mood, mode, top_n=6)
```

No changes needed to your recommendation engine!

---

## Next Steps

1. **Start the server**: `python flask_app.py`
2. **Open browser**: http://localhost:5000
3. **Select your mood** and enjoy!

The frontend is fully functional and connected to your existing recommendation system. All the beautiful UI elements are working with your real data!

🎉 **Enjoy your new beautiful frontend!** 🎉
