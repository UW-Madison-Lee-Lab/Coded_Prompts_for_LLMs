<h1 align="center"> <p>Coded Prompts for Large Language Models</p></h1>
<h4 align="center">
    <p><a href="https://myhakureimu.github.io/" target="_blank">Ziqian Lin</a>, <a href="https://www.linkedin.com/in/yicong-chen-046993250/" target="_blank">Yicong Chen</a>, <a href="https://yzeng58.github.io/zyc_cv/" target="_blank">Yuchen Zeng</a>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a></p>
    <p>University of Wisconsin-Madison</p>
    </h4>

**Paper Link**: TBD

While Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks and various prompting techniques have been proposed, there remains room for performance enhancement. In this work, we introduce a novel dimension to prompt design – coded prompts for LLM inference. Drawing inspiration from coding theory, where coded symbols communicate or store functions of multiple information symbols, we design coded prompts to process multiple inputs simultaneously. We validate this approach through experiments on two distinct tasks: identifying the maximum prime number within a range and sentence toxicity prediction. Our results indicate that coded prompts can indeed improve task performance. We believe that coded prompts will pave a new way for innovative strategies to enhance the efficiency and effectiveness of LLMs.

## Experiment

### Task 1: Finding the Maximum Prime Number in a Range (Binary Classification)
step 1 - install the openai package: 
    
    pip install openai

step 2 - download the code for this repo and unzip it: 

step 3 - cd to the folder: 

    cd task\_1

step 4 - run experiment: 

    python run_task1.py --integers 1 --samples 4 --apikey yourkey

### Task 2: Online Comment Toxicity Prediction (Regression)
