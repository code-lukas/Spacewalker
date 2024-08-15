run command:
`docker compose -f docker-compose.yml -f docker-compose-develop.yml up --remove-orphans --force-recreate`

parcel:
- open devcontainer
- `cd frontend`
- `npx parcel ./src/index.html --dist-dir=/workspaces/SpaceWalker/backend/static/frontend`

Upon saving, changes to the frontend code are applied
