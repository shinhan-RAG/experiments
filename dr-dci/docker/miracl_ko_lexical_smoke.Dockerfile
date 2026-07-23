FROM python:3.12-slim-trixie@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless=21.0.11+10-1~deb13u2 \
    && ln -s "/usr/lib/jvm/java-21-openjdk-$(dpkg --print-architecture)" "${JAVA_HOME}" \
    && rm -rf /var/lib/apt/lists/*

# Pyserini declares model-serving dependencies too.  This smoke pins only the
# import closure needed by its Lucene sparse index/search API; no model package
# or external model/API client is installed or called.
RUN python -m pip install --no-cache-dir --no-deps pyserini==2.1.0 \
    && python -m pip install --no-cache-dir \
        numpy==2.4.2 \
        pandas==2.3.3 \
        pyjnius==1.7.0 \
        tqdm==4.67.1

WORKDIR /work
