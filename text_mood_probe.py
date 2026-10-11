"""
Audit phrases for the "Describe your day" text mood detector.

A small, fixed, hand-written probe set used by evaluate_text_mood.py and
tests/test_text_mood.py to check the production inference path
(src.ml_inference.predict_text_mood). It is NOT training data: none of these
sentences may appear in text_vocab.py or data/text_mood_dataset.csv
(a test enforces this), so results on it are not inflated by memorisation.

It is also NOT an authoritative benchmark. The phrases were written for this
audit (2026-10-10) by the project, they are short and clean, and the
"expected" labels are one reasonable reading of each sentence. Treat results
as a smoke test of obvious behaviour, not as a measure of real-world
accuracy.

Each entry: (text, acceptable moods, kind)
  kind "clear"     -- one mood is the natural reading
  kind "mixed"     -- more than one mood is a fair reading; any listed is acceptable
  kind "neutral"   -- no real emotional content; an honest detector should say it is unsure
"""

PROBE = [
    # ---- happy ----
    ("I love my work.", {"happy"}, "clear"),
    ("I am proud of what I accomplished today.", {"happy"}, "clear"),
    ("Today was a really good day.", {"happy"}, "clear"),
    ("I just passed my driving test!", {"happy"}, "clear"),
    ("Dinner with old friends made me so happy.", {"happy"}, "clear"),
    ("I'm grateful for everything I have.", {"happy"}, "clear"),
    ("My team won the match and I can't stop smiling.", {"happy"}, "clear"),
    ("I really enjoy spending time with my family.", {"happy"}, "clear"),
    ("Got great feedback from my manager.", {"happy"}, "clear"),
    ("My sister had her baby and everyone is thrilled.", {"happy"}, "clear"),
    # ---- sad ----
    ("I feel lonely and sad.", {"sad"}, "clear"),
    ("I miss my grandmother so much.", {"sad"}, "clear"),
    ("Nobody remembered my birthday.", {"sad"}, "clear"),
    ("I feel empty and nothing makes me happy anymore.", {"sad"}, "clear"),
    ("My best friend moved away and I feel lost.", {"sad"}, "clear"),
    ("I cried myself to sleep last night.", {"sad"}, "clear"),
    ("It's been a gloomy, heavy kind of day.", {"sad"}, "clear"),
    ("I feel like nobody understands me.", {"sad"}, "clear"),
    ("We had to put our cat down today.", {"sad"}, "clear"),
    ("I failed the exam I studied so hard for.", {"sad"}, "clear"),
    # ---- angry ----
    ("I am furious about what happened.", {"angry"}, "clear"),
    ("My coworker took credit for my work again.", {"angry"}, "clear"),
    ("I'm so annoyed that they cancelled without telling me.", {"angry"}, "clear"),
    ("Someone keyed my car and I'm livid.", {"angry"}, "clear"),
    ("I can't stand how rude he was to me.", {"angry"}, "clear"),
    ("This is so unfair, I want to scream.", {"angry"}, "clear"),
    ("I'm sick of being ignored in meetings.", {"angry"}, "clear"),
    ("The landlord kept my deposit for no reason and I'm fuming.", {"angry"}, "clear"),
    ("Stop lying to me!", {"angry"}, "clear"),
    ("I'm really mad at my brother right now.", {"angry"}, "clear"),
    # ---- chill ----
    ("I feel peaceful and relaxed.", {"chill"}, "clear"),
    ("Just sipping tea and watching the rain.", {"chill"}, "clear"),
    ("Lazy Sunday on the couch, no plans.", {"chill"}, "clear"),
    ("Taking it slow today, everything is calm.", {"chill"}, "clear"),
    ("Sitting by the beach listening to the waves.", {"chill"}, "clear"),
    ("I'm unwinding after a long week with a good book.", {"chill"}, "clear"),
    ("Feeling mellow and content tonight.", {"chill"}, "clear"),
    ("Nothing to do and I'm totally fine with that.", {"chill"}, "clear"),
    ("Enjoying a quiet evening at home.", {"chill"}, "clear"),
    ("A long warm bath has me completely at ease.", {"chill"}, "clear"),
    # ---- energetic ----
    ("I am excited and full of energy.", {"energetic"}, "clear"),
    ("Just crushed my morning run, let's go!", {"energetic"}, "clear"),
    ("I'm pumped for the concert tonight.", {"energetic"}, "clear"),
    ("Ready to take on the world today!", {"energetic"}, "clear"),
    ("I can't sit still, I have so much energy.", {"energetic"}, "clear"),
    ("Let's dance all night!", {"energetic"}, "clear"),
    ("Feeling motivated and ready to get things done.", {"energetic"}, "clear"),
    ("Big game today and I'm fired up.", {"energetic"}, "clear"),
    ("Woke up early and I'm buzzing.", {"energetic"}, "clear"),
    ("Hit a new personal record at the gym.", {"energetic"}, "clear"),
    # ---- stressed ----
    ("I feel exhausted and overwhelmed.", {"stressed"}, "clear"),
    ("I am worried about my deadlines.", {"stressed"}, "clear"),
    ("I have three exams tomorrow and haven't studied.", {"stressed"}, "clear"),
    ("My inbox is out of control and I can't keep up.", {"stressed"}, "clear"),
    ("I'm anxious about the job interview.", {"stressed"}, "clear"),
    ("Bills are piling up and I don't know how to pay them.", {"stressed"}, "clear"),
    ("There's too much on my plate right now.", {"stressed"}, "clear"),
    ("I can't sleep because I keep thinking about work.", {"stressed"}, "clear"),
    ("My boss keeps adding tasks and I'm drowning.", {"stressed"}, "clear"),
    ("I'm so tense, my shoulders are in knots.", {"stressed"}, "clear"),
    # ---- mixed / ambiguous: any listed mood is a fair reading ----
    ("I love my work but I'm exhausted.", {"happy", "stressed"}, "mixed"),
    ("I got the job but I'm nervous about starting.", {"happy", "stressed"}, "mixed"),
    ("Excited but a little scared about moving.", {"energetic", "stressed", "happy"}, "mixed"),
    ("Bittersweet goodbye at my farewell party.", {"sad", "happy"}, "mixed"),
    ("I'm tired.", {"sad", "stressed", "chill"}, "mixed"),
    ("Can't wait for the weekend!", {"happy", "energetic"}, "mixed"),
    ("My deadline moved and now I can finally relax.", {"chill", "happy"}, "mixed"),
    ("The traffic was awful but the concert was amazing.", {"happy", "energetic", "angry"}, "mixed"),
    ("Busy day at work.", {"stressed", "energetic"}, "mixed"),
    # ---- neutral: no emotional content ----
    ("It's Monday.", set(), "neutral"),
    ("Work.", set(), "neutral"),
    ("I had a sandwich for lunch.", set(), "neutral"),
]

# A second set, written on 2026-10-10 BEFORE the 2026-10-10 retraining and
# never used to choose training examples or thresholds. PROBE above was
# inspected while diagnosing the model, so it is effectively a development
# set; BLIND is the fairer (still small and hand-written) check.
BLIND = [
    ("Spent the afternoon laughing with my cousins.", {"happy"}, "clear"),
    ("Finally got my visa approved, what a relief and a joy.", {"happy"}, "clear"),
    ("My painting got picked for the school exhibition.", {"happy"}, "clear"),
    ("Everything about this trip has been wonderful.", {"happy"}, "clear"),
    ("I adore my new apartment.", {"happy"}, "clear"),
    ("Woke up to a sweet message from my partner and I'm smiling.", {"happy"}, "clear"),
    ("I keep thinking about the friends I lost touch with and it hurts.", {"sad"}, "clear"),
    ("The rain matches how hopeless I feel.", {"sad"}, "clear"),
    ("My dad is in hospital and I feel so low.", {"sad"}, "clear"),
    ("I didn't get into the university I wanted and I'm heartbroken.", {"sad"}, "clear"),
    ("I sat alone at lunch again today.", {"sad"}, "clear"),
    ("Grief comes in waves and today it's a big one.", {"sad"}, "clear"),
    ("The delivery guy threw my package over the fence, I'm so angry.", {"angry"}, "clear"),
    ("They raised the rent again and I'm outraged.", {"angry"}, "clear"),
    ("I hate being talked down to.", {"angry"}, "clear"),
    ("My flatmate ate my food again and I'm seething.", {"angry"}, "clear"),
    ("Why does nobody ever listen, this makes my blood boil.", {"angry"}, "clear"),
    ("Got cut off in traffic and I'm still irritated.", {"angry"}, "clear"),
    ("Watching the sunset from the balcony, totally serene.", {"chill"}, "clear"),
    ("Slow morning with coffee and soft music.", {"chill"}, "clear"),
    ("Just floating in the pool without a care.", {"chill"}, "clear"),
    ("Meditated for twenty minutes and feel tranquil.", {"chill"}, "clear"),
    ("Cozy blanket, candle on, nowhere to be.", {"chill"}, "clear"),
    ("A gentle walk in the park cleared my head.", {"chill"}, "clear"),
    ("Can't wait to hit the dance floor tonight!", {"energetic"}, "clear"),
    ("Three espressos in and I'm ready to go go go.", {"energetic"}, "clear"),
    ("Training for the marathon and feeling strong.", {"energetic"}, "clear"),
    ("So hyped for the festival this weekend!", {"energetic"}, "clear"),
    ("Jumped out of bed full of drive today.", {"energetic"}, "clear"),
    ("Let's get this workout started!", {"energetic"}, "clear"),
    ("Presentation tomorrow and I'm a nervous wreck.", {"stressed"}, "clear"),
    ("So many assignments due, my head is spinning.", {"stressed"}, "clear"),
    ("I'm panicking about money this month.", {"stressed"}, "clear"),
    ("Juggling two jobs is wearing me down.", {"stressed"}, "clear"),
    ("The move, the kids and work, it's all too much.", {"stressed"}, "clear"),
    ("I keep checking my email in a cold sweat.", {"stressed"}, "clear"),
    ("Going to the shop later.", set(), "neutral"),
    ("Tuesday.", set(), "neutral"),
]
