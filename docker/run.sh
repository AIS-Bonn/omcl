docker run -P --net host --runtime=nvidia --gpus all -it --rm -v ~/data:/home/omcl_user/data \
    -v ./src/omcl:/home/omcl_user/omcl/omcl \
    -v ./data_scripts:/home/omcl_user/omcl/data_scripts \
    -v .cache/:/home/omcl_user/omcl/.cache \
    -v ./third_party/:/home/omcl_user/omcl/third_party \
    omcl
