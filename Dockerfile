FROM toxicode/stable-diffusion:base

COPY server /workspace/dream-server

# RUN wget -nc -O /workspace/sd.zip https://pixeldrain.com/u/yD33e2mE && \
#     unzip /workspace/sd.zip -d /workspace/dream-server/models && \
#     rm /workspace/sd.zip
#
# RUN wget -nc -O /workspace/dream-server/models/inpainting_last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1


WORKDIR /workspace/dream-server


ADD docker_start.sh /start.sh
RUN chmod a+x /start.sh

CMD [ "/start.sh" ]