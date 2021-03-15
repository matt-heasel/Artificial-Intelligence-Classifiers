classdef myknn
    methods(Static)
        
        function m = fit(train_examples, train_labels, k)%only require 3 argumanets to call
            %applies z-score standardisation by default
            %k = number of nearest neighbours
            
            % start of standardisation process
			m.mean = mean(train_examples{:,:});%call to mean 
			m.std = std(train_examples{:,:});%call to standard deviation
            %both used to rescale feature values to lie within a similar
            %range, prevents noisy data effecting results and feature with
            %larger values (ie measured in cm instead of mm from drowning
            %out potentially useful patterns i the data)
            for i=1:size(train_examples,1)%loops over all training examples applying z-score standardisation
				train_examples{i,:} = train_examples{i,:} - m.mean;%subtraction of mean so that the feature distribution is centred on 0
                train_examples{i,:} = train_examples{i,:} ./ m.std;%divides by the standard deviation of the feature
                %after this z-score standardisation process all feature
                %distributions will have the same mean value of zero
                %and the same "spread", a standard deviation of 1.
                %this is desriable to notice the trends in the data without
                %noisy or extreme values drowning out smaller feature
                %values.
            end
            % end of standardisation process
            
            m.train_examples = train_examples;%fit takes copy of all training exmaples after standardisation
            %this is stored in structure m so that the identical process
            %can be applied to and compared to the testing data in the
            %prediction function.
            m.train_labels = train_labels;%fit takes copy of all training labels
            m.k = k;% number of nearest neighbours we want to use is copied to field of structure m called k
        
        end

        function predictions = predict(m, test_examples)%accepts as many exmaples as needed in a table test_examples

            predictions = categorical;

            for i=1:size(test_examples,1)%loops through all of the test examples,
                %applies standardisation to each one
                %calling the predict one function on every example
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1));
                
                this_test_example = test_examples{i,:};%pulls one test example outof table used later to add to end of predictions
                
                % start of standardisation process
                this_test_example = this_test_example - m.mean;%same standardisation process applied to testing data as to trainng data in the fit function
                %
                this_test_example = this_test_example ./ m.std;
                % end of standardisation process
                
                this_prediction = myknn.predict_one(m, this_test_example);%calls predict_one function on current example
                predictions(end+1) = this_prediction;%adds prediction of this_test_example to the predictions function
            
            end
        
		end

        function prediction = predict_one(m, this_test_example)
            
            distances = myknn.calculate_distances(m, this_test_example);%calls calculate distance function to find the distance between
            % this_test_example (the one test example) and all of the
            % testing examples that were copied into (classifier) structure m at the end
            % of the fit method
            neighbour_indices = myknn.find_nn_indices(m, distances);%passess strucure m and distances array to neigbour indeces (which is an array)find_nn_indeces function
            %returns the index of the k nearest distances to this function
            %as neighbour_indices,
            prediction = myknn.make_prediction(m, neighbour_indices);%neighbour_inideces array is k nearest distances 
        
        end

        function distances = calculate_distances(m, this_test_example)%arguments are classifier structure and the test example comparing to them
            
			distances = [];
            
			for i=1:size(m.train_examples,1)%loops through training examples, to allow a comparison of this_test_example against all the training examples from m
                %loop used to find the total amount of training examples to
                %iterate over by calling size of classifier traing examples
                %for all rows 
				this_training_example = m.train_examples{i,:};%each iteration of the loop, a new training example is read and assigned to 
                %this_training_example
                this_distance = myknn.calculate_distance(this_training_example, this_test_example);%so the classifier can calculate the distance between the training 
                %and testing data 
                distances(end+1) = this_distance;%adds each new distance between the initially passed test example and the 
                %next training example to the distances array
            end
        
		end

        function distance = calculate_distance(p, q)%takes current training example (out of all of the ones stored in structure m)
            %and the testing example
            %this function is called every iteration of the loop in the
            %predictions function, so every element from the training data
            %will be inserted into array p and compared to every element in
            %array q which is the testing data, the arithmetic below will
            %be called for every combination of elements from the 2 arrays
            %********************************************************************************************************************************
            
			differences = q - p;%matlab automatically subtracts two arrays element by element
            %so the elemnt in array p is from the training data array and
            %the current testing example is in array q which is subtracted
            %from p
            squares = differences .^ 2;%square all the differences, including the " . " means matlab squares the differences element by element 
            total = sum(squares);%add up all te numbers in the square array
            distance = sqrt(total);%find the square root of the total 
        
		end

        function neighbour_indices = find_nn_indices(m, distances)%the distances array from the predict one function is passed in
            %which is then sorted and assigned an index to keep track of
            %which distance corresponds to which row of data(which example)
            
            %creates 2 arrays, sorted, which contains the sorted distances
            %between the training and testing example, and indices which
            %contain the corresponding index of that distance (which
            %row/exmaple it corresponds to in the table)
			[sorted, indices] = sort(distances);%depending on what index the distance had in the original array will correspond to the 
            %row in the table it belings to (4th index in distances = 4th
            %row in table)
            neighbour_indices = indices(1:m.k);%returns first k indexs of sorted distances array(indeces) that correspond
            %to the rows of data in structure m.k, then returns them to
            %predict_one as neighbour_indeces
        
		end
        
        function prediction = make_prediction(m, neighbour_indices)%looks up neighbour_indices corresponding class labels in structure m
            %and calculates the mode from the k nearest neighbour, this
            %"prediction" is KNN theory in practice

			neighbour_labels = m.train_labels(neighbour_indices);%training labels for neighbour_indeces(which is the array of k nearest neighbours indices 
            %stored in neighbour_labels array which is then passed to the
            %mode function and stored in prediction array
            prediction = mode(neighbour_labels);
        
		end

    end
end

