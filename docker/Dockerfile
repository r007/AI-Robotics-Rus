# syntax=docker/dockerfile:1

# Build stage: mdBook with the plugins used by the book, then the merged site.
FROM rust:1-slim-bookworm AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN cargo install --locked mdbook@0.5.4 mdbook-mermaid@0.17.1 mdbook-katex@0.10.0
WORKDIR /src
COPY . .
RUN rm -rf build site && python3 scripts/build_site.py

# Serve stage: static files only.
FROM nginx:1-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/site /usr/share/nginx/html
EXPOSE 80
