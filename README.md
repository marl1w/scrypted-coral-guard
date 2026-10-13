# Coral Guard

Verify a camera's motion with a Google Coral Edge TPU before it reaches your automations, recordings and notifications.

The repository has two parts:

- [coral-verifier/](coral-verifier/README.md): a small HTTP service (packaged as a Home Assistant add-on) that owns the Coral USB stick and runs a YOLOv9 detector. Clients post JPEG frames per camera; it tracks people, vehicles and animals across frames and says when one is confirmed, still present or gone. The HTTP contract is [coral-verifier/API.md](coral-verifier/API.md).
- [scrypted-plugin/](scrypted-plugin/README.md): a Scrypted plugin that sits on each camera's motion sensor, sends frames to the service when the camera reports motion (and a few idle frames to keep tracking), and in Gate mode publishes motion only when the service confirms something. It fails open: if the service is unavailable, the camera's own motion passes through.

## How they fit

The plugin talks to the service over HTTP, by default on port 8750, with a shared token sent as `X-Verifier-Token`. The service is never in the video path; it only sees the frames the plugin posts.

## Install

1. Service: in Home Assistant, add this repository to the add-on store (Settings, Add-ons, Add-on store, Repositories: `https://github.com/marl1w/scrypted-coral-guard`), install Coral Verifier, set its `token` option and start it. Without Home Assistant, build `coral-verifier/Dockerfile` and run the container with the Coral USB device passed through (see its README).
2. Plugin: build and install the Scrypted plugin from `scrypted-plugin/` (see its README), enter the service URL and token on the plugin page, then enable Coral Guard per camera under Extensions.

## Documentation

- Plugin: [scrypted-plugin/README.md](scrypted-plugin/README.md), [scrypted-plugin/docs/DEVELOPMENT.md](scrypted-plugin/docs/DEVELOPMENT.md)
- Service: [coral-verifier/README.md](coral-verifier/README.md), [coral-verifier/API.md](coral-verifier/API.md), licences and attribution in [coral-verifier/NOTICE.md](coral-verifier/NOTICE.md)
