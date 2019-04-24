FROM conda/miniconda3

# update conda
RUN conda update -n base -c defaults conda

# add environment
ADD environment.yml /tmp/environment.yml

# create virutal environment
RUN conda env create -f /tmp/environment.yml

# activate environment
RUN echo "source activate iasi" > ~/.bashrc

# add binaries to path
ENV PATH /opt/conda/envs/iasi/bin:$PATH

WORKDIR /iasi