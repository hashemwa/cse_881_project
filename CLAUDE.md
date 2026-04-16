
### Project Context
Project Title: Automated Classification of Human vs AI Postings
Difficulty Level: Hard (5 points)
The breakdown of criterion is given below.
• Data Collection (+1 point)
o Scrape job and agricultural web postings
• Ground Truth Labelling (+0.5 points)
o Collected postings will be manually labeled as human
o AI generated postings will be manually labeled as AI
• Data Preprocessing (+1 point)
o Text cleaning, stopwords removal, n-gram extraction using NLTK
o Salary and location normalization for job postings
o Add labels and a target column.
• Algorithm Development (+1 point)
o Designing and implementing our own AI text detection algorithm.
• Evaluation (+0.5 points)
o Evaluation of algorithms/model architecture will be compared to baselines
o Evaluation of algorithms/models on varying, distinct datasets
• Prototype Development (+1 point)
o Interactive front-end for user interactivity and model performance
Abstract:
The goal of this project is to develop a system to accurately classify and detect AI
generated text across online postings. To do this, we will scrape human generated data
from job and agricultural postings. Fake postings will be generated from LLMs or collected
from pre-existing datasets. This text data will then be cleaned, preprocessed, and labeled.
Classification of posts will involve classical machine learning methods, neural networks,
and LLMs. An interactive Streamlit platform will also be built and deployed to display
our classification model results and performances.
Data Collection:
The first source of human generated text was collected from indeed.com. The first dataset
contains data science job postings. This includes job titles, salaries, locations, and the full
job description. The scraped data is stored as a csv file which is then preprocessed. We
used octoparse (https://www.octoparse.com/) as the tool to scrape the job data. The tool
has a UI to browse the pages to be scraped and to select the desired data. It then visits the
links on that page and collects the data which we then exported as a csv file.
Since the listings have different properties, for example, some have the salary listed in the
description, while some have it listed in a field that is directly scraped by the tool. Another
example is how locations are listed in the listings, with some of them having the state
name and some having the state abbreviation. Some preprocessing was done to help our
machine learning model deal with the data. The final dataset would have four main fields:
• Job title: containing the title of the job posting.
• Job location: in which we will need to make it a unified format.
• Job salary: which we are using regular expression to extract it from the full
description if not scraped.
• Job summary: containing the full job description.
This dataset is then combined with AI-generated postings. We collected the postings from
multiple AI sources to have a variety of AI styles and compare them to human postings. The
data was collected from the following AI sources:
• Claude AI (https://claude.ai/)
• Microsoft Copilot (https://copilot.microsoft.com/)
• ChatGPT (https://chatgpt.com/)
• Google Gemini (http://gemini.google.com)
• Perplexity AI (https://www.perplexity.ai/)
The combined dataset has about 2,000 entries, and we made sure to shuffle the data
within the dataset to avoid having any order in the data that the machine learning model
might use to learn the data.
For our second source of human-generated text data, we collected farm and agricultural
listings from the Care Farming Network Directory (https://carefarmingnetwork.org/find-a-
care-farm/). These listings consist of farms, farmers’ markets, producers, and many other
farming models across the United States.
Before collecting the data, we first inspected the website’s crawling policy via its robots.txt
file to ensure compliance with their data scraping policies. After inspection, we found no
limitations or restrictions against data scraping. In addition, we found the domain’s XML
sitemap, which could be used to gather all the URLs for individual member farm listings.
From here, we developed a python script to navigate to each URL and scrape the relevant
text data. This was done by utilizing playwright for browser automation and beautifulsoup
for parsing to scrape the relevant text data.
After scraping, we end up with 390 individual listings stored in a JSON format with the
following key attributes:
• id: a unique identifier for each listing, generated from the URL slug
• url: source URL of the individual listing
• name: name of the listing
• description: detailed description of the listing and its products/practices
An additional attribute, label, was added to each entry of the JSON file. This class attribute
represents the ground truth label and was set to “Human” for all 390 records, indicating
that the data was real human generated text.
For the AI-generated text data on agricultural listings, we utilize the NVIDIA NIM (NVIDIA
Inference Microservices) API, which is a platform that provides free, pre-built, and
optimized generative AI models for deployment. To ensure that the generated data
followed the same schema as the human data, we developed a detailed system prompt to
guide the models on text generation. This prompt is given below:
You are an expert directory copywriter and an authentic voice in the sustainable agriculture
and care farming community. Your task is to generate a realistic, fictional profile for a care
farm, producer market, or agricultural community in the United States.
Content Guidelines:
• Niche Focus: Ensure the profile reflects the true nature of a "care farm." It should
mention therapeutic agriculture, populations served (e.g., veterans, individuals with
developmental disabilities, youth at risk, or those in addiction recovery), and
community integration alongside standard farming practices.
• Agricultural Details: Include specific, realistic details about the farm's operations
(e.g., specific crops, livestock breeds, regenerative practices, CSA programs).
• Tone and Style: The tone must be authentic, warm, and grounded. Avoid overly
polished, generic, or "marketing-speak" language. Write as if the farm owner or a
local community member wrote it. Use varied sentence structures.
• Length: Length: The description should be a detailed block of text (roughly 50 to 250
words). Format the description as a single continuous string, avoiding bulleted lists.
Output Format Constraints:
Provide the output STRICTLY as a single, valid JSON object. Do not include any
conversational text, pleasantries, or explanations before or after the JSON. Do not wrap the
output in Markdown code blocks (e.g., do not use ```json).
The JSON must contain the exact following keys:
- “id”: a unique, lowercase, hyphen-separated, URL-friendly slug based on the
listing’s name
- “url”: a fictional URL formatted exactly as
"https://carefarmingnetwork.org/directory-member_farms/listing/[id]/"
- “name”: the name of the fictional listing
- “description”: a detailed, multi-sentence paragraph explaining the listing’s mission,
practices, community involvement, and agricultural produces
This prompt was passed to four powerful instruction-tuned large language models
(LLMs), each generating 100 unique listings, for a total of 400 AI-generated records.
The models selected were:
• gpt-oss-120b
• glm-5
• gemma-3-1b-it
• qwen2.5-7b-instruct
Similarly to the human dataset, an additional class attribute, label, was appended to each
entry and set to “AI”, designating them as synthetically generated text. Finally, we
combined the 390 authentic human records with the 400 AI listings, creating a single
dataset consisting of agricultural listings.
Data Preprocessing:
For the job postings dataset, we need to preprocess the postings to extract some useful
information and fill out missing fields based on the information that is included in the job
posting. So far, we have completed a script that parses the text of the job posting and
extracts the salary information and the location information from the job posting, as
sometimes these pieces of information are only available in the job description, and we
need to have them in their own fields. Moreover, sometimes the format is different. For
example, some salaries are posted as annual salaries, and some are hourly. We need to
make sure that we have the same format to make the data more consistent. This script will
be used on both human-collected data and AI-generated data. We will also need to make
sure to include that same preprocessing script on the data that the user enters later on to
have consistency in our data.
Another preprocessing method that we will implement is using NLTK (Natural Language
Toolkit) to help further process the data. The plan is to remove stopwords, probably
lowercase everything, and then extract n-grams from the job descriptions. This will give us
more insight into the data, and we want to compare how machine learning models perform
on the data before and after using NLTK. This will also become handy when we implement
our algorithm that aims to manually check for different attributes that are present in AI-
generated text.
For the agricultural listings, preprocessing followed a similar structure to the job data.
First, we applied standardization steps across the description attribute. This involved
HTML/noise removal, where any residual HTML tags or markdown formatting were stripped
from the scraped human data. We also transformed the target variable into binary
encoding, where “Human” was mapped to 0 and “AI” mapped to 1.
Finally, we partitioned the preprocessed dataset into training, validation, and testing sets
using an 80-10-10 split. For this splitting, we applied stratified sampling based on the
encoded labels to maintain a balanced number of human and AI instances across all three
splits.
Timeline:
- April 6 – April 11: Finish all data preprocessing including NLTK processing on both
datasets. Train and evaluate baseline models (logistic regression, random forest,
etc.). Begin developing custom AI detection algorithm.
- April 12 – April 18: Finish custom AI detection algorithm and neural network models.
Run final evaluations across all datasets. Build the Streamlit app and perform tests
to make sure everything is working as expected.
- April 19 – April 20: Create and finalize the presentation.
- April 21 or April 23: Presentation due.
- April 23 – April 24: Review and finalize project code, results, and report.
- April 24: Final report due.
Team Members and Contributions:
Hussain Aljafer: Collected job posting data and combined the jobs dataset. Generated
some AI job postings from different AI models. Implemented some data preprocessing
functions to be used for the jobs dataset (extracting salaries, locations, and formatting
locations and salaries) and to be used for the other datasets. Implementing baseline
classification models, proposing and helping brainstorm and design the AI detection
algorithm, helped writing this report.
Wahid Hashem: Collected AI-generated job postings using Claude Opus model to add to
the human job posting data. Will contribute to data preprocessing to prepare the text for
our machine learning models. Will design and build the Streamlit app, including UI/UX, to
allow users to interact with the models and see how they perform. Will contribute to writing
the project report and creating the presentation.
Aryan Sharma: Utilized ChatGPT’s o3 Deep Research and Google Gemini’s 3 Flash Thinking
to generate AI job postings for inclusion in the human job posting data. Time permitting, I
will create a third task that differentiates between Human and AI social media posts (most
likely Reddit) and follow a similar methodology as the tasks explained above. Plan on
assisting with the research and development of baseline classification machine learning
models for effective comparison to our AI algorithm in the evaluation stage. Once in the
final stage, I will communicate relevant points in the report and presentation.
Ricky Li: Scraped human created agricultural listings. Developed script to generate AI
listings. Helped with preprocessing and cleaning functions. Help implement baseline
models for classification. Work on implementing more models/techniques for detection
using neural networks / large language models. Help create final presentation and report.