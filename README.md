# yt-chat

Un lien YouTube. Un transcript horodaté.
Pose tes questions. Codex répond avec les passages sources.

## Ce qui change

`yt-chat` devient un simple skill Codex : le script récupère les sous-titres publics sans clé API YouTube, puis Codex s'appuie sur ce transcript pour répondre aux questions. L'application Chainlit, les modèles configurés, Qdrant, Docker et Poetry ont été retirés. Aucune synthèse automatique n'est imposée avant les questions.

## Installation

Copier ce dépôt dans `~/.codex/skills/yt-chat`, puis installer l'unique dépendance Python dans un environnement virtuel :

```bash
python3 -m venv ~/.codex/skills/yt-chat/.venv
~/.codex/skills/yt-chat/.venv/bin/python -m pip install -r ~/.codex/skills/yt-chat/requirements.txt
```

Le skill peut ensuite être invoqué avec `$yt-chat` ou choisi automatiquement pour une question sur une vidéo YouTube. Les réponses utilisent le modèle déjà disponible dans Codex ; aucune clé OpenAI propre à `yt-chat` n'est nécessaire.

## Utilisation directe du script

```bash
~/.codex/skills/yt-chat/.venv/bin/python ~/.codex/skills/yt-chat/scripts/get_transcript.py 'https://www.youtube.com/watch?v=jNQXAC9IVRw' --output /tmp/yt-chat-transcript.md
```

Le script accepte aussi un identifiant vidéo, un lien `youtu.be`, `shorts` ou `live`. Il privilégie les sous-titres français puis anglais et, à défaut, prend une piste disponible dans sa langue d'origine. Utiliser `--languages en,fr` pour changer l'ordre. La sortie Markdown conserve les horodatages et des liens directs vers les passages.

La récupération dépend des sous-titres accessibles publiquement et de l'accès de YouTube depuis la machine. Une vidéo sans sous-titres, privée, inaccessible ou bloquée peut échouer. Le script ne transcrit pas l'audio lui-même.
