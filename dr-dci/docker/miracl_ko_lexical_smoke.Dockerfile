FROM python:3.12-slim-trixie@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless=21.0.11+10-1~deb13u2 \
    && mkdir -p /opt/java \
    && ln -s "/usr/lib/jvm/java-21-openjdk-$(dpkg --print-architecture)" "${JAVA_HOME}" \
    && rm -rf /var/lib/apt/lists/*

# The Pyserini distribution carries the pinned Anserini fat JAR.  Run that JAR
# directly: this avoids importing Pyserini's dense/impact modules and installs
# neither model frameworks nor external model/API clients.
RUN python -m pip install --no-cache-dir --no-deps pyserini==2.1.0

WORKDIR /work
