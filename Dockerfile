FROM conda/miniconda3

WORKDIR /iasi

# add environment
ADD environment.yml /iasi/environment.yml

# add setup script
ADD script/setup.sh /iasi/script/setup.sh

# run setup secipt
RUN /bin/bash script/setup.sh

# add binaries to path
ENV PATH /opt/conda/envs/iasi/bin:$PATH
