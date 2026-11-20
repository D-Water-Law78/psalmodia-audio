# Calage des paroles : Whisper + Deepgram + MMS

Outils utilisés pour les versions « Deepgram », « Deepgram 2 » et « Mixte » de la catégorie Test.

1. **Deepgram** (`deepgram_caler.py`) : transcription par Deepgram (modèle `whisper-large`, en grec et en arabe ;
   les modèles Nova ne reconnaissent pas le chant). Le copte est lu avec la phonétique arabe, puis calé mot à mot.
   La clé se passe dans l'en-tête `Authorization: Token …` de l'appel à `api.deepgram.com/v1/listen` ; elle n'est jamais écrite ici.
   Limite : Deepgram ne donne pas de mot de plus de 0,5 s, donc aucune vocalise.
2. **MMS** (`mms_caler.py`) : calage forcé de Meta (torchaudio `MMS_FA`), copte écrit en latin avec la prononciation
   copto-arabe, arabe romanisé par `uroman`. Donne un temps pour chaque lettre.
   `python mms_caler.py essai-fil/index.html ID audio.m4a sortie.json emission.pt`
3. **Fusion** (`fusion.py`) : vote mot par mot entre la version actuelle, Deepgram et MMS ; là où deux sources concordent
   (< 0,6 s) on pose une ancre ; entre deux ancres, MMS recale chaque lettre dans cette fenêtre seulement ;
   dans un passage sans paroles et sans ancre, on garde la version actuelle.
   `AUDIO=audio.m4a python fusion.py essai-fil/index.html ID deepgram.json deepgram_mask.json mms.json emission.pt sortie.json`
