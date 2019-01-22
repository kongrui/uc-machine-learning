#

## Submission No.1

### Definition

- Definition
  An important part of preparing for a project is conducting some form of literature review. 

  This gives you a sense of what work has been done in the field, what techniques were used, and what kind of performance you might expect to see. 

  Is there any interesting academic research that has been published on this topic? If so, it should be cited here. If not, make a note of that. Again, the point is mainly just to demonstrate that you have gone through this step

- Problem Statement
  In this section you should also give a brief overview of your approach to the problem, in terms of what techniques you're planning to use. Of course, you don't need to go into great detail on how they work (that'll come later), but at least mentioning things by name gives your reader a good idea of what to expect in the coming sections. It acts as a nice summary

- Metrics
  In this section it's required that you give a mathematical definition for your metric in terms of its calculation

  It's also required that you justify your choice based on its characteristics and the characteristics of your desired model. What makes this metric the optimal choice here?

### Analysis

- The source of the data, its size, the features and their structure should be all clearly described, which will make for a solid overview of the key characteristics

  Some descriptive statistics of these features should be summarized as well, or at least the interesting ones. How are these things distributed? Feature distribution is a critical characteristic that influences models greatly, so it should be given proper attention here

  Data visualization, The visualizations should also be accompanied by a discussion of the main points and an analysis that tells the audience what we can learn from looking at them
  
- **Visualization** These are important data qualities and certainly things well worth visualizing; good use of this section here
  
  The visualizations themselves are clean and well presented, with appropriate labels and identifiers, and the right visual encoding for each data type
  
  The visualizations should also be accompanied by a discussion of the main points and an analysis that tells the audience what we can learn from looking at them  

- **Algorithm**, In this section, you should focus on describing how your algorithms work, in terms of how they train and predict. What's the procedure behind the scenes? What's the theory behind it? This discussion doesn't have to be mathematically rigorous, unless that suits you better; essentially, however you can best explain these concepts, this is how you should approach it
  
  As an example, if you were describing the SVM, you might discuss such concepts as 'maximizing the margin' and the 'kernel trick'

  The purpose of taking this theoretical perspective is that it helps your reader to understand how it's treating the data, and therefore gives more objective reasoning for why it may be optimal in this situation
  
  It also clearly demonstrates that you have the fundamental mathematical understanding of machine learning algorithms that would be expected of an engineer, and this will help your portfolio


### Methodology

- The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.

  The goal of this section is to make our work as reproducible as possible; for any future researchers that read your work and wish to expand on it, they'll have to start by re-implementing what you have done, and they can only do that if your explanation of your work through this report is detailed and accurate. So more of a step by step walkthrough of the process of building these models should be prepared, with this perspective in mind. Would someone reading your work be able to replicate it?

  In addition, one of the main ways we help with reproduciblity is by clearly documenting the challenges we faced, and how we overcame them. This helps out those that are following our work not to get stuck on the things that we got stuck on. If you can include some discussion of any coding complications you faced in this process, and how you overcame these, that will round out this section nicely. If nothing particularly difficult happened and it was straightforward, discuss why this was the case

  The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.

  Here it's required that you provide the hyperparameters tuned, the values tried, and the results obtained. This will fully characterize the process and make it reproducible

### Results
- The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.

  Your final results are presented in a way that's easy to analyze and compare, with good surrounding discussion
  
  Another important aspect of this section is discussing robustness. Does your model generalize well to unseen data? Is it sensitive to small changes in the data, or to outliers? Essentially, can we trust this model, and why or why not? You should attempt to answer this question in the most objective way you can, ideally with some well formed tests


### Conclusion
  Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.

  Solid recap of the overall process
  
  It's also important to reflect on your learning process through this project. This will help you to internalize what you learned better. Ultimately, this is the primary goal of the capstone, as it is designed as a learning experience. So to do this, consider such questions as: what was the most interesting aspect of this work? What was the biggest challenge, and how did you overcome it? What did you learn?