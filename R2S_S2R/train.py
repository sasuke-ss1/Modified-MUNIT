from data_loader import dataLoader

def train(model, dataloader, epochs):
    dataset = dataloader.load_data()
    dataset_size = len(dataloader)

        for epoch in range(epochs):

            total_error = None
            avg_error = None

            for i, data in enumerate(dataset):
                
                model.set_input(data)
                model.optimize_parameters()
                errors = None#errors = model.get_current_errors()

                if total_error == None:
                    #total_error = errors
                    #avg_error = errors
                    pass
                else:
                    for k in errors:
                        total_error[k] += errors[k]
                

