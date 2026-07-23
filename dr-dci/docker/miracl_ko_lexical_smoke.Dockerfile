FROM python:3.12-slim-trixie@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless=21.0.11+10-1~deb13u2 \
    && mkdir -p /opt/java \
    && ln -s "/usr/lib/jvm/java-21-openjdk-$(dpkg --print-architecture)" "${JAVA_HOME}" \
    && rm -rf /var/lib/apt/lists/*

# The Pyserini source distribution carries the pinned Anserini fat JAR. Verify
# its source hash, extract only that JAR, then remove Pyserini itself. This
# avoids importing its dense/impact modules or retaining model/API dependencies.
# NumPy supports the repository's existing paired-bootstrap evaluator only.
ARG PYSERINI_SOURCE_SHA256=384fb783c52ac1605caabe8a75f520323dfed2b5595072911c87f6cfca8bf15f
RUN mkdir -p /tmp/pyserini-source /opt/anserini \
    && python -m pip download --no-cache-dir --no-deps --dest /tmp/pyserini-source pyserini==2.1.0 \
    && test "$(sha256sum /tmp/pyserini-source/pyserini-2.1.0.tar.gz | awk '{print $1}')" = "$PYSERINI_SOURCE_SHA256" \
    && python -m pip install --no-cache-dir --no-deps /tmp/pyserini-source/pyserini-2.1.0.tar.gz \
    && cp /usr/local/lib/python3.12/site-packages/pyserini/resources/jars/anserini-2.1.1-fatjar.jar /opt/anserini/ \
    && python -m pip uninstall -y pyserini \
    && python -m pip install --no-cache-dir --no-deps numpy==2.4.2 \
    && rm -rf /tmp/pyserini-source

WORKDIR /work
