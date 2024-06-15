# fromstring 에러
fromstring이 deprecated 됨
```
DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
```
solution
```
Replacing 'fromstring' with 'frombuffer'
```

# matplotlib 경고
main thread 밖에서 실행됨
```
UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
```
solution
```py
import matplotlib
matplotlib.use('agg')
```

# GCP, cv2 오류1
lib.so.1 file이 없음 (로컬 시스템은 존재하지만, 도커 컨테이너에는 누락된 종속성)
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
solution
```
conda remove opencv
pip3 install opencv-python-headless
```

# GCP, GLIBCXX_3.4.29 오류
opencv, torch에서 발생, 해당 파일을 가져올 수 없음
```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/juchan/miniconda3/envs/crowded/lib/python3.9/site-packages/torch/lib/libtorch_python.so) 
```
solution
```
sudo apt install binutils                                                               # strings 사용하기 위함
strings /home/juchan/miniconda3/envs/crowded/lib/libstdc++.so.6 | grep GLIBCXX          # GLIBCXX_3.4.29 존재 확인

sudo rm /lib/x86_64-linux-gnu/libstdc++.so.6
sudo cp /home/juchan/miniconda3/envs/crowded/lib/libstdc++.so.6 /lib/x86_64-linux-gnu   # libstdc++ 복사
```

# GCP, conda 에러
conda, pip 이것저것 만지다가 발생
```
# >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

    Traceback (most recent call last):
      File "/home/juchan/miniconda3/lib/python3.12/site-packages/conda/exception_handler.py", line 17, in __call__
        return func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
 ...

An unexpected error has occurred. Conda has prepared the above report.
If you suspect this error is being caused by a malfunctioning plugin,
consider using the --no-plugins option to turn off plugins.

Example: conda --no-plugins install <package>

Alternatively, you can set the CONDA_NO_PLUGINS environment variable on
the command line to run the command without plugins enabled.

Example: CONDA_NO_PLUGINS=true conda install <package>

If submitted, this report will be used by core maintainers to improve
future releases of conda.
Would you like conda to send this report to the core maintainers? [y/N]: 
Timeout reached. No report sent.
```
solution
```
conda clean -all        # 캐시 삭제
```