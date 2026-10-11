# MoodCast listening pilot (Phase 2L.1) -- protocol

Pilot comparing two music rankings for four "Match my mood" requests (happy,
sad, chill, energetic). Each request is a *pair* (P1-P4) of two 8-track lists,
labelled A and B. The labels and orders are random; nothing on the listening
sheet says which ranking produced a list.

## Roles
- **Administrator** prepares the sheet, collects ratings and alone holds
  `data/metadata/listening_pilot_2l1/answer_key/`. The administrator must not
  tell listeners anything about how lists were made, and must not open the
  answer key until every rating is in.
- **Listeners** receive only `listening_sheet.csv` and the three rating
  templates (`ratings_tracks.csv`, `ratings_lists.csv`, `ratings_pairs.csv`).

## Blinding: what it does and does not cover
- The sheet shows no strategy, seed, rank or score, and track order inside each
  list is shuffled.
- It is **not** fully blind: listening on Spotify shows each track's title,
  artist and artwork, and a listener may recognise songs. If the administrator
  is also a listener, or has seen the answer key or the generating code, the
  study is not blind for that person -- record this in the notes.
- The current ranking removes speech-like tracks from calm lists, so content
  differences (e.g. spoken-word tracks) can themselves hint at a list's origin.
  That is part of what is being measured, not a leak in the materials.

## Listening procedure (per listener)
1. Use the same device, headphones/speakers and a comfortable fixed volume for
   the whole session. Close other audio.
2. Work through pairs in sheet order (P1 to P4). Within a pair, listen to list
   A then list B, tracks in sheet order.
3. For each track, play about 30 seconds from the start (skip a silent intro if
   needed; use the same rule throughout). Do not read lyrics or reviews.
4. Rate each track right after listening, in `ratings_tracks.csv`:
   mood fit to the request (1 = not at all, 5 = perfectly), enjoyment (1-5),
   and whether it is mostly speech rather than music (yes/no).
5. After each list, rate it in `ratings_lists.csv`: overall mood fit, perceived
   variety (1 = all sounds the same, 5 = very varied while still fitting), and
   enjoyment (1-5).
6. After both lists of a pair, record the preferred list (A, B or none) in
   `ratings_pairs.csv`. "None" means no real preference.
7. Put your listener id (any code, no names) in the first column of every row
   you fill. Leave a cell blank if you could not rate it (e.g. a link that does
   not play) and say why in notes.
8. Do not change earlier ratings after hearing later lists.

## Links and metadata
Links are Spotify track pages built from track ids (no API calls). Titles and
artists come only from the optional local metadata file; where a track has none,
the sheet says "unavailable" -- nothing is guessed.
