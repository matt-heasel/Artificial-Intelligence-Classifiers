classdef mynb%naive bayes cant capture dependent relationships, the relationship between two dependent features is lost
    %uses normal probability function for each feature distribution,
    %made up o its own mean and standard deviation,each feature is the
    %considered in isolation
    methods(Static)
        
        function m = fit(train_examples, train_labels)%m structure created with training examples and labels
            
            m.unique_classes = unique(train_labels);%unique function called to find each of the potential class labels
            m.n_classes = length(m.unique_classes);%length called on the result to find how many unique classes there are

            m.means = {};
            m.stds = {};
            
            for i = 1:m.n_classes%looping over all the examples of training data
            
				this_class = m.unique_classes(i);% set up the this_class variable as an example:
                examples_from_this_class = train_examples{train_labels==this_class,:};%gets all the exampls data that corresponds to the current class
                %(this_class) in training labels array, from the training
                %examples array
                m.means{end+1} = mean(examples_from_this_class);%standard deviation and mean calculated
                m.stds{end+1} = std(examples_from_this_class);%stores each element of data in a cell array (can store anything even another array)
                %stored in the m structure m.means for the mean values and
                %m.stds for the standard deviations
            
			end
            
            m.priors = [];%prior is an estimate of how likely each class label is to occur based on how many times it is in the training data
            
            for i = 1:m.n_classes%second loop for the training data, going over all the values in the m structure (uses the m.n_classess, 
                %which is the number of unique classes calculated in the fit function previously
                
				this_class = m.unique_classes(i);%% set up the this_class variable as an example:
                examples_from_this_class = train_examples{train_labels==this_class,:};%pulls examples of data from the corresponding class labels
                m.priors(end+1) = size(examples_from_this_class,1) / size(train_labels,1);%stores in priors array, the calculation of the likelihood that
                %a random example selected from the training example will
                %belong to this class
            
			end

        end

        function predictions = predict(m, test_examples)

            predictions = categorical;

            for i=1:size(test_examples,1)%loops through all the test examples

				fprintf('classifying example %i/%i\n', i, size(test_examples,1));%to get each row of data we need the height of the table thus use of size
                this_test_example = test_examples{i,:};%sets this_test_example as currently selected data example
                this_prediction = mynb.predict_one(m, this_test_example);%calls predict one function on currently seleceted data
                predictions(end+1) = this_prediction;%adds result of above code to predictions array
            
			end
        end
%The predict_one() function loops over all the possible class labels and, for each one, calculates a 
%likelihood for the current test example given the class by calling the calculate_likelihood() function.
        function prediction = predict_one(m, this_test_example)%function loops over all the class labels, calculating the likelihood for the current test
            %example.
            % It then computes a value which is proportional to the posterior: the probability of the current class label, given the test example.
            %It does this by multiplying the likelihood it has just calculated with the prior for the current class.

            for i=1:m.n_classes%loops over class labels to allow the likelihood function to be called on each one

				this_likelihood = mynb.calculate_likelihood(m, this_test_example, i);
                this_prior = mynb.get_prior(m, i);%get prior just gets the prior for the current class
                posterior_(i) = this_likelihood * this_prior;%calculates posterior value based on percentage of previous class labels whi
            
            end
        %we're really interested in is the index of the array element containing the maximum value, because this tells us which element of
        %unique_classes to look in for the name of the predicted class label. 
            [winning_value_, winning_index] = max(posterior_);%so we ask for the second value returned from the max function, the winning index
            prediction = m.unique_classes(winning_index);

        end
        
        function likelihood = calculate_likelihood(m, this_test_example, class)%calculate likelihood function,considers the value of every feature 
            %in the current test example.
            % likelihood is a prime example of the class-conditional
            % independence assumption of Naive Bayes in action.
			likelihood = 1;
            
			for i=1:length(this_test_example)
                likelihood = likelihood * mynb.calculate_pd(this_test_example(i), m.means{class}(i), m.stds{class}(i));%calculates
                %the probability density for each feature value
                %calls its own calculate_pd function
                % multiplying together probability densities for each feature in the example
            end
        end
        
        function prior = get_prior(m, class)
            
			prior = m.priors(class);%just gets the prior from the structure
        
		end
        
        function pd = calculate_pd(x, mu, sigma)
        
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
        
		end
            
    end
end