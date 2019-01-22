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
  
  
## Submission No.2

### Algorithm

Algorithms and techniques used in the project are thoroughly discussed and properly justified based on the characteristics of the problem.

You have explained the approach you are going to take to face the problem and you have mentioned the algorithms that you are going to use. However, it's important to demonstrate that you have a solid base of the algorithms that you're using. In particular, it's expected more details, at least, of the most important algorithms in your report. Please, be sure to explain the main hyper-parameters for each algorithm too (at least those used in the Refinement section).

I see in your report (section "Train out-of-box models") that you are using LR, DT, Gradient Boosting and RF, in some of them like in Gradient Boosting there is a layman's term explanation of the model. That could be a good example of what is needed to include (here in this section or even in that section if you want to keep the structure of your current document, although I think is better to keep the theoretical part before the implementation) for the rest of the algorithms.

Just after that section we can find the "Refinement" section where you are naming some features that the reader won't understand unless an explanation is included. This is the main reason for this section: to put some background for the rest of the report, something like: "these are the algorithm I have decided to use (...) and they work like this (...) and these are the most important hyper parameters (...) that we are going to use to refine them".

As you can see with all that information, the reader will have a good picture and background to continue understanding the document.

### Implementation

The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.

I understand that your intention in this section is to explain the general strategy followed to solve the problem and I think this is needed and very informative, however, in this section is important to focus your attention on how you have implemented the algorithms. For example, you could explain the steps followed by a piece of code or an explanation about how you have implemented your solution. The idea is to explain your code to someone that has not read it, with enough detail to understand it (you could focus your explanation on the most important parts of your code).
Note

If I had to do this, I'd move the theoretical part to the section "algorithms" and keep the implementation here. Please, keep in mind that this section is where we are expecting to see how you have implemented the solution, so is a good idea to include some code, is not needed but is the easier way to explain the current implementation of your solution.


### Process of improvement

The process of improving upon the algorithms and techniques used is clearly documented. 

I understand that your intention in this section is to explain the general strategy followed to solve the problem and I think this is needed and very informative, however, in this section is important to focus your attention on how you have implemented the algorithms. For example, you could explain the steps followed by a piece of code or an explanation about how you have implemented your solution. The idea is to explain your code to someone that has not read it, with enough detail to understand it (you could focus your explanation on the most important parts of your code).
Note

If I had to do this, I'd move the theoretical part to the section "algorithms" and keep the implementation here. Please, keep in mind that this section is where we are expecting to see how you have implemented the solution, so is a good idea to include some code, is not needed but is the easier way to explain the current implementation of your solution.


### Results

The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.

Please be sure to also use your results to make an argument for why your final model represents a reasonably robust result for this problem using the features/parameters of the model.
Please let me use this definition to help you to improve this section.
"The robustness of a model is the property that characterizes how effective your algorithm is while being tested on the new independent (but similar) dataset. In the other words, a model robust is the one which does not deteriorate too much when training and testing with slightly different data (either by adding noise or by taking other dataset). here". For that reason to test the robustness it's needed to use your model with more than a single dataset and study the result to see that there is a coherence between the results.

If I had to demonstrate the robustness of a model I'd use sklearn KFolds and will study the score for each fold checking that the results are coherent between them. With this, you can demonstrate that your model was not lucky with the data used and is robust enough.


### Conclusion

A visualization has been provided that emphasizes an important quality about the project with thorough discussion. Visual cues are clearly defined.

Please, try to provide a section including a visualization about the results of your model or about something you would like to highlight about your model.
