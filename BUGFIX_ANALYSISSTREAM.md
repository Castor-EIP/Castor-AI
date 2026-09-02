# Bug: AnalysisStream reste sourd au flux client une fois l'analyse démarrée

## Fichier concerné

`server/src/castostudio_ai_server/service.py` — méthode `AnalysisStream`.

## Résumé

Depuis le hotfix `0730081` ("syntaxe error while starting ia worker", 8 juillet),
`AnalysisStream` imbrique la boucle d'analyse continue à l'intérieur de la
boucle de lecture des messages client :

```python
async for request in request_iterator:
    if payload == "sources":
        session.sources = self._convert_sources(...)
        yield status_event
        async for event in self._analysis_loop(session):   # <-- bloque ici
            yield event
    elif payload == "stop":
        ...
```

`_analysis_loop` tourne `while session.context.session_id in self._sessions`,
donc indéfiniment tant que la session n'est pas supprimée. Tant que cette
boucle tourne, le `async for request in request_iterator` externe est
**suspendu** — le serveur ne relit plus jamais le flux entrant.

## Impact concret

Une fois qu'une session a envoyé son premier `SourceList` :

- Tout nouveau `SourceList` envoyé ensuite (URLs rafraîchies, metadata
  `is_speaking`/`active_speaker` mise à jour en live) **n'est jamais lu**.
  Le module IA (podcast, football, tout module) analyse indéfiniment les
  mêmes `sources` figées reçues au tout premier message.
- Un `StopSignal` envoyé dans le flux devient du code mort : il ne sera
  jamais traité tant que l'analyse tourne. Seul `EndSession` (RPC unaire
  séparé) permet encore de couper une session.
- `KeepAlive` n'est plus non plus traité une fois l'analyse démarrée.

Concrètement : si un client s'attend à pouvoir renvoyer des `SourceList` en
continu (pattern courant pour pousser des niveaux audio / statut speaker en
direct), le serveur ignore tout ça après le premier message. Ça a un effet
direct sur la pertinence de n'importe quel module d'analyse — pas seulement
podcast.

## Repro

`tests/test_service.py::test_stream_sources_returns_scene_switch` (avant fix)
bloque indéfiniment avec un module qui renvoie une décision à chaque cycle,
car plus aucun `StopSignal` n'est jamais lu pour terminer le flux.

## Fix appliqué

Lecture du flux client et boucle d'analyse tournent maintenant en tâches
asyncio **concurrentes**, reliées par une `asyncio.Queue` d'événements
sortants :

- `read_client_messages()` : lit `request_iterator` en continu, met à jour
  `session.sources` sur chaque nouveau `SourceList`, pousse les events
  status/error/stop dans la queue. Ne bloque plus jamais sur l'analyse.
- `_run_analysis_loop(session, outgoing)` (ex-`_analysis_loop`, renommée) :
  tourne indépendamment, lit `session.sources` à chaque itération — capte
  donc naturellement les mises à jour poussées par `read_client_messages`
  puisque c'est le même objet `_Session` partagé (mutable).
- La boucle principale de `AnalysisStream` consomme la queue et `yield` les
  events au client gRPC, jusqu'au sentinel `None` (flux client terminé) ou
  fermeture du stream.
- Sur sortie (normale ou exception), les deux tâches sont annulées
  proprement (`task.cancel()` + `await` pour absorber le `CancelledError`).

Comportement observable : un `StopSignal` envoyé pendant que l'analyse
tourne est maintenant traité immédiatement (dans les ~100ms du prochain tick,
pas jamais). Un `SourceList` mis à jour est pris en compte dès la prochaine
itération d'analyse (~100ms), plus figé pour toute la durée de la session.

## Tests

Les deux tests qui exerçaient `AnalysisStream` avec un module "toujours
actif" (`test_stream_sources_returns_scene_switch`,
`test_stream_sources_logs_flow_and_masks_source_values`) supposaient un
seul event en retour — hypothèse qui ne tenait déjà plus avec la boucle
d'analyse infinie introduite par le hotfix, indépendamment du bug de
concurrence (ils plantaient/bloquaient déjà avant ce fix). Réécrits pour
envoyer un `StopSignal` après le `SourceList` (comme un vrai client) et
vérifier qu'un switch arrive puis que le `STOPPED` final est bien reçu.

Suite complète : `uv run pytest` → 20/20 verts.

## Pour vérifier en live

Lancer une session, envoyer un `SourceList`, puis pendant que l'analyse
tourne, envoyer un second `SourceList` avec des metadata différentes (ex.
`is_speaking` qui change) : la décision suivante doit refléter le nouveau
contenu sans attendre la fin de session. Avant ce fix, elle restait basée
sur le tout premier message.
