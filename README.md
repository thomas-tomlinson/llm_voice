tom's LLM learning
==================

this is a osx python specific learning project.  The goal was to produce something that could do the following:
* utilize apples MLX library to do apple silicon native AI inteferrance 
* preform STT (Speech to Text) and input that into a LLM chat style
* perform TTS (Text to Speech) to speak the LLM generated response.
* learn more about how all of these components work


How to use this
-----------------

1. install ollama (https://ollama.com)
1. once ollama is installed, pull down the llama3.1 image.  `ollama pull llama3.1:latest`
2. clone this repo
3. create a conda local environment.  For example `conda create -p ./.conda python=3.12` 
4. activate the conda environment.  `conda activate ./.conda`
5. install the required packages with `pip install -r requirements.txt`
6. run `python realtime_whispher.py`

The first time you run this, there will be a lot of downloads involved so be patient.  Once it's ready you'll
be prompted to his the enter key to start recording.  There's a wake word of "computer" that triggers the STT process and 
after a very poorly implemented "no more talking", the prompt will be submitted to the LLM.  There's some debugging still in
place that you are of course free to disable / remove / change etc.

Lessons Learned thus far
----------------------------
* how cow, "detecting" voice audio is much harder than i thought.
* my math skills for numpy leave a lot to be desired.  
* This guys whisper MLX implementation is amazing: https://github.com/mustafaaljadery/lightning-whisper-mlx
* even non MLX, ollama is pretty damn cool.
* I had no idea that apple had some many additional voices that work under the `say` command (https://support.apple.com/en-gb/guide/mac-help/mchlp2290/mac#:~:text=and%20speaking%20rate.-,add%20a%20new%20voice,-You%20can%20add)

Directions I may learn next
--------------------------
* implement a proper VAD (Voice Activity Detection) algorithm.  I'd like to avoid the blanket 2 second sleep to try and catch an actual phrase.
* explore some other TTS solutions.
* summarize and prune the chat history.  right now it just endlessly appends which is not a great long term solution.