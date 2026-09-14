FROM ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740

RUN python -m pip install --no-cache-dir --force-reinstall --no-deps evalplus==0.3.1 \
    && python -c "import importlib.metadata as m; assert m.version('evalplus') == '0.3.1'"
