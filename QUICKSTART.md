# Quick start

1. Install the dependencies (Python 3.13):

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Start the server:

   ```bash
   python flask_app.py
   ```

   On Windows you can double-click `start_moodcast.bat` instead.

3. Open http://127.0.0.1:5000.

## What you'll see

- **Home**: pick one of six moods (happy, sad, angry, chill, energetic,
  stressed), or describe your mood in a sentence. Then choose "Match my mood"
  or "Cheer me up".
- **Music**: a ranked list of tracks. Each track links to its page on
  Spotify; no audio plays inside MoodCast. This needs the locally built music
  catalog. Without it, the page explains what's missing.
- **Movies**: needs `data/movies_final.csv`, which is not distributed. Without
  it, the page explains that movie recommendations are unavailable.

To enable music recommendations, follow "Music catalog" in
[README.md](README.md#music-catalog-optional).

## Troubleshooting

- **Port 5000 in use:** set `FLASK_PORT=5001` in `.env` (copy it from
  `.env.example`) or in the environment.
- **"Music recommendations are unavailable":** build the catalog with
  `python train_music_278k.py` after downloading the 278k dataset (see
  [DATA.md](DATA.md)).
- **"Movie recommendations are unavailable":** expected. The movie dataset is
  not distributed (see [DATA.md](DATA.md#movie-data)).
- **`No module named ...`:** run `python -m pip install -r requirements.txt`.
