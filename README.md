# Research-and-Analysis-of-Speech-Enhancement-or-Dereverberation (RA-SED)
This repository contains some material of speech enhancement and dereverberation. On the one hand, I summarize this work for my further understanding. On the other hand, I hope that all beginners or masters interested in speech enhancement can ask me questions and make progress together.  
A lot of my summary is not very good, I hope you put forward corrections!  

<b>Advertisement：</b>  
I would like to open source a speech enhancement toolkit in the near future, but there is currently no good way to do frame-level feature extraction. I would like to put the features in one file, but currently running on a small memory machine while reading and writing may run out of memory.  
If you have a better way, please contact me!  
Thank you!  

`
My email: hshi.cca@gmail.com, hshi_cca@tju.edu.cn (I will not be able to use this email after Jan. 2021!)
`

## 0. Outlines
> 1. Overviews
>> 1.1 What is speech enhancement (dereverberation)  
>> 1.2 Classification of speech enhancement (dereverberation)  
> 2. Traditional speech enhancement or dereverberation methods (I will show this part in the future.)  
> 3. Deep learning-based speech enhancement or dereverberation methods
>> 3.1 Basic framework and classification  
>> 3.2 Frequency domain speech enhancement (dereverberation)
>>> 3.2.1 Strategy of research
>>> 3.2.2 
> 4. Public datasets  
> 5. Performance  
> 6. Future trends  
> 7. Acknowledge  


## 1. Overviews
We will give some basic introduction in this part. We first introduce what is speech enhancement (dereverberation) and its mathematical expression. Then we will give the classification of speech enhancement (dereverberation) which we summarized. 

### 1.1 What is speech enhancement (dereverberation)
In real life, microphone pickup, in addition to receiving voice, will also receive some noise and reverberation. Speech enhancement is aimed at noisy speech, want to get clean speech. But in fact, speech enhancement (dereverberation) will bring some distortion of noise signal, and can't restore clean speech. 
`Speech enhancement (dereverberation) is speech noise reduction (denoising).`  

The mathematical expression is as follows:  
$x = r * s + n$  
$s$ is speech signal (desired), $r$ is room impulse response (RIR), $n$ is additive noise signal, and $x$ is microphone pickup signal, the noisy signal. 
The speech enhancement system wants to recover $s$ from $x$.  











