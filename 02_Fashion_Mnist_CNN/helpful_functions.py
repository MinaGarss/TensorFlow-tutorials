import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_single_graph(train_history, valid_history, metric_name):
    ## visualizing the loss among different epochs
    ## using an interactive graph

    ## creating a figure object
    fig = go.Figure()

    ## adding the training loss graph to the plot
    fig.add_trace(go.Scatter(x0=1, dx=1,
                             y=train_history,
                             mode='lines',
                             name=f"Training {metric_name.title()}",
                             marker={'color':"red"},
                             hovertemplate = 'Epoch = %{x}<br>'+
                             f'Training {metric_name.title()} '+'= %{y:.4f}<extra></extra>',
                             hoverlabel={'bgcolor':'rgba(255,0,0,0.5)'}))

    ## adding the validation loss graph to the plot
    fig.add_trace(go.Scatter(x0=1, dx=1,
                             y=valid_history,
                             mode='lines',
                             name=f"Validation {metric_name.title()}",
                             marker={'color':"green"},
                             hovertemplate = 'Epoch = %{x}<br>'+
                             f'Validation {metric_name.title()} '+'= %{y:.4f}<extra></extra>',
                             hoverlabel={'bgcolor':'rgba(0,255,0,0.25)'}))

    ## updating the layout of the graph by adding titles and labels
    fig.update_layout(title=f'The {metric_name.lower()} value VS the number of epochs',
                      xaxis_title='Number of epochs')
    
    ## updating the y axis title according to the metric name
    if metric_name.lower() == 'loss':
        fig.update_layout(yaxis_title='Loss "the lower the better"')
    else:
        fig.update_layout(yaxis_title='Accuracy "the higher the better"')

    ## extending the x axis range by 1 from the left and the right
    fig.update_xaxes({'range':[0, len(train_history)+1]})

    ## creating buttons to change the hover behavior
    my_buttons = [{'label': "unlinked", 'method': "update", 'args': [{}, {"hovermode": 'closest'}]},
                  {'label': "linked", 'method': "update", 'args': [{}, {"hovermode": 'x'}]}]

    ## adding the created buttons to the plot and setting their position
    fig.update_layout({
        'updatemenus':[{
            'type': "buttons",
            'direction': 'left',
            'pad':{"l": 0, "t": 0},
            'active':0,
            'x':0,
            'xanchor':"left",
            'y':1.1,
            'yanchor':"top",
            'buttons': my_buttons}]})

    ## showing the final plot
    fig.show("notebook")


def plot_double_graph(train_loss, valid_loss, train_accuracy, valid_accuracy):

    ## combining the 2 graphs above into 1 graph
    ## creating a figure object
    fig = make_subplots (rows=1, cols=2, 
                         # Set the subplot titles
                         subplot_titles=['Loss', 'Accuracy'],
                         # Add spacing between the subplots, ranges from 0 to 1
                         horizontal_spacing=0.15)


    ## adding the training loss graph to the plot
    fig.add_trace(go.Scatter(x0=1, dx=1,
                             y=train_loss,
                             mode='lines',
                             name="Training Loss",
                             marker={'color':"red"},
                             hovertemplate = 'Epoch = %{x}<br>'+
                             'Training Loss = %{y:.4f}<extra></extra>',
                             hoverlabel={'bgcolor':'rgba(255,0,0,0.5)'}),
                  row=1, col=1)

    ## adding the validation loss graph to the plot
    fig.add_trace(go.Scatter(x0=1, dx=1,
                             y=valid_loss,
                             mode='lines',
                             name="Validation Loss",
                             marker={'color':"green"},
                             hovertemplate = 'Epoch = %{x}<br>'+
                             'Validation Loss = %{y:.4f}<extra></extra>',
                             hoverlabel={'bgcolor':'rgba(0,255,0,0.25)'}),
                  row=1, col=1)

    ## adding the training accuracy graph to the plot
    fig.add_trace(go.Scatter(x0=1, dx=1,
                             y=train_accuracy,
                             mode='lines',
                             name="Training Accuracy",
                             marker={'color':"red"},
                             hovertemplate = 'Epoch = %{x}<br>'+
                             'Training Accuracy = %{y:.4f}<extra></extra>',
                             hoverlabel={'bgcolor':'rgba(255,0,0,0.5)'}),
                  row=1, col=2)

    ## adding the validation accuracy graph to the plot
    fig.add_trace(go.Scatter(x0=1, dx=1,
                             y=valid_accuracy,
                             mode='lines',
                             name="Validation Accuracy",
                             marker={'color':"green"},
                             hovertemplate = 'Epoch = %{x}<br>'+
                             'Validation Accuracy = %{y:.4f}<extra></extra>',
                             hoverlabel={'bgcolor':'rgba(0,255,0,0.25)'}),
                  row=1, col=2)


    ## updating the layout of the graph by adding titles and labels
    fig.update_layout(xaxis_title='Number of epochs',
                      xaxis2_title='Number of epochs',
                      yaxis_title='Loss "the lower the better"',
                      yaxis2_title='Accuracy "the higher the better"')

    ## extending the x axis range by 1 from the left and the right
    fig.update_xaxes({'range':[0, len(train_loss)+1]})

    ## creating buttons to change the hover behavior
    my_buttons = [{'label': "unlinked", 'method': "update", 'args': [{}, {"hovermode": 'closest'}]},
                  {'label': "linked", 'method': "update", 'args': [{}, {"hovermode": 'x'}]}]

    ## adding the created buttons to the plot and setting their position
    fig.update_layout({
        'updatemenus':[{
            'type': "buttons",
            'direction': 'left',
            'pad':{"l": 0, "t": 0},
            'active':0,
            'x':0,
            'xanchor':"left",
            'y':1.2,
            'yanchor':"top",
            'buttons': my_buttons}]})

    ## showing the final plot
    fig.show("notebook")


def plot_images(images, class_names, true_labels, model_probs):

    ## calculate the number of rows required for our plots
    n_rows = int(np.ceil(len(images)/2))

    ## create the required number of subplots
    fig, axs = plt.subplots(nrows=n_rows, ncols=4, figsize=(16,4*n_rows))

    ## counters to help assign the graph to the axes
    row_counter = 0
    column_counter = 0

    ## define the styling colors
    correct_color = '#327346'
    wrong_color = '#b32520'
    default_color = '#2163bf'

    for i in range(len(images)):

        ## assigning the true label and the predicted class to variables
        true_label = true_labels[i]
        model_predictions = model_probs[i]
        predicted_class = model_predictions.argmax()
        
        ## specifing which axis are we working on for image plotting
        img_ax = axs[row_counter][column_counter]

        ## removing tick marks
        img_ax.set_xticks([])
        img_ax.set_yticks([])

        ## displaying the photo
        img_ax.imshow(images[i], cmap=plt.cm.binary)

        ## setting the text beneath the photo
        img_ax.set_xlabel(f'{class_names[true_label]}', color=default_color, fontsize=12)
        
        ## incrementing the column to switch to the other axis
        column_counter += 1
        
        ## specifing which axis are we working on for probability distribution
        prob_ax = axs[row_counter][column_counter]
        
        ## plotting the probability distribution
        prob_ax.bar(x=range(len(class_names)),height=model_predictions, color=default_color)
        prob_ax.bar(true_label, height=model_predictions[true_label], color=correct_color)
        
        if true_label == predicted_class:
            prob_ax.set_xlabel(f'{class_names[true_label]}  {100*model_predictions[true_label]:.2f}%',
                               color=correct_color, fontsize=12)
        else:
            prob_ax.bar(predicted_class, height=model_predictions[predicted_class],
                        color=wrong_color)
            prob_ax.set_xlabel(f'{class_names[predicted_class]}  {100*model_predictions[predicted_class]:.2f}%',
                               color=wrong_color, fontsize=12)

        ## incrementing the row and column counter for the next image
        column_counter += 1
        if column_counter == 4:
            column_counter = 0
            row_counter +=1
    
    plt.show()


def get_report(model, data, labels, class_names):
    
    ## get the model's predictions
    predictions = model.predict(data)

    ## putting the results into a Pandas DataFrame
    results = pd.DataFrame()
    results['True_Class'] = labels
    results['Predicted_Class'] = predictions.argmax(axis=1)
    results['Is_Correct'] = results.True_Class == results.Predicted_Class

    ## aggregating through the DataFrame to build a report
    report = results.groupby('True_Class').agg(
        Percent_Correct = ('Is_Correct', 'mean'),
        Num_Correct = ('Is_Correct', 'sum'),
        Num_Total = ('Is_Correct', 'count'))

    ## adding the class names to the report
    report.insert(0, 'Class_Name', class_names)

    return report