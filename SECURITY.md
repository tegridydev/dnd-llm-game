# Security policy

The supported release is **v0.4.1**. Please share security reports privately through the repository’s Security tab. Include the version, impact, and steps that demonstrate the problem with data you control.

## Deployment

Run on loopback on a single user machine. The API checks origins and hosts to protect against unrelated websites, and has no multi-user authentication. Keep it behind the local firewall and off public proxies or tunnels.

Model credentials belong in the server environment. Choosing a hosted narrator sends campaign context and retrieved lore passages to that provider. Remote embedding requires a separate opt-in because indexing sends PDF text to the embedding service. Remote model connections require HTTPS and TLS 1.2 or newer. Model requests do not follow redirects.

PDF ingestion runs in a separate process with a deadline. Linux also enforces a worker address-space limit. The worker runs as the local user. Uploads, save files and the lore index should be kept under the user’s private data directory.

The app never executes model-generated commands. Narrative and utility output are treated as data, and the game engine owns dice rolls and combat calculations.
