# SpottySearch
 
Find a piece of pottery by describing it. A small multimodal search tool for a paint-your-own pottery café.
 
## Background
 
Spotty Pots is a family-run paint-your-own pottery café. When a customer paints a piece, a staff member photographs it before it is glazed and fired, and posts the photo to a WhatsApp group labelled with the customer's name. When the customer returns to collect the finished piece, the photo is used to find it.
 
That lookup falls apart when the name is forgotten or misspelt, or the photo has scrolled out of reach. SpottySearch lets staff find a piece by describing it instead, for example "a white mug with a caterpillar", and returns the closest matches.
 
## How it works
 
Two complementary retrieval routes run over the same set of photos.
 
**Image search (CLIP).** Every photo is encoded with CLIP (ViT-B/32) into an embedding. A text query is encoded by the same model and ranked against the image embeddings by cosine similarity. This route is strongest on overall colour, shape and style.
 
**Caption search (VLM + text embeddings).** Every photo is described by a vision-language model (Qwen2.5-VL-3B-Instruct), which lists each item and its colour, form and any patterns. Those captions are embedded with a sentence model (all-MiniLM-L6-v2) and searched the same way. This route is strongest on specific painted text and named designs.
 
The two routes are independent and can be compared, or their scores fused. CLIP is a contrastive, discriminative model; the captioning step is generative, which is what produces the searchable descriptions.
 
## Features
 
- Natural-language search over pottery photos, returning the top matches with each image and its description.
- Runs entirely locally on Apple Silicon (MPS), so no data leaves the machine and there are no API costs.
- Generated captions cope with multi-item photos and read painted text on the pottery.
- Caption generation is cached and resumable, so an interrupted run picks up where it left off.
## Tech stack
 
Python, PyTorch, OpenAI CLIP, Qwen2.5-VL via Hugging Face Transformers and qwen-vl-utils, sentence-transformers, NumPy, Pillow, matplotlib.
 
## Setup
 
Python 3.11 or 3.12 is recommended.
 
```bash
pip install torch pillow numpy matplotlib
pip install git+https://github.com/openai/CLIP.git
pip install transformers qwen-vl-utils accelerate
pip install sentence-transformers
```
 
Qwen2.5-VL needs a recent version of Transformers, so update it if the model fails to load.
 
## Usage
 
The pipeline is currently driven from a Jupyter notebook, in these stages:
 
1. Point the code at a folder of photos.
2. Build the CLIP image index (embed every photo once and cache it).
3. Generate a description per photo with Qwen2.5-VL (cached to `captions.json`).
4. Build the caption index (embed every caption with the sentence model).
5. Search by text and view the top matches with their descriptions.
## Data
 
The sample photos are not committed. The real café images contain customers' names and personalised designs, so they are kept private. Point the code at your own image folder to try it.
 
## Evaluation
 
Each of 20 hand-labelled photos provides a search query (its written description), with that same photo as the correct result. Both methods rank all 43 photos for every query, and I report recall@k and mean reciprocal rank (MRR) over the labelled queries.
 
| Method | recall@1 | recall@3 | recall@5 | MRR | Median rank |
| --- | --- | --- | --- | --- | --- |
| CLIP (image) | 0.45 | 0.65 | 0.70 | 0.60 | 2 |
| Caption (Qwen + text) | 0.60 | 0.80 | 0.80 | 0.71 | 1 |
 
Caption search is the stronger route on this set, returning the correct pot first 60% of the time and within the top three 80% of the time, for a median rank of 1. CLIP holds its own on pots described by overall colour and shape, but is weaker when a query leans on painted text or names, which the captioning model transcribes and the text search can match directly.
 
The caption route's main failure mode is cluttered photos. When a scene contains background objects, such as other items on the table or shelves, the 3B vision-language model sometimes describes those as pottery too, which pollutes the caption and pushes the correct match down the ranking. One such photo fell to rank 26. Cropping or segmenting the piece before captioning, or prompting the model to ignore the background, is the obvious mitigation.
 
These figures come from only 20 queries, so they are indicative rather than precise: a single pot moving in or out of the top three shifts recall@3 by five points.
 
## Roadmap
 
- Reduce background clutter in captions by cropping or segmenting the piece before captioning, the main cause of poor matches in the evaluation.
- Hybrid caption search (BM25 plus embeddings) to catch exact terms such as names painted on a piece.
- Score fusion across the image and caption routes.
- A simple web app to replace the WhatsApp workflow, with a proper record per piece and search built in.
## Status
 
Personal prototype and work in progress.