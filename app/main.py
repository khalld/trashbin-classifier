import gradio as gr
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from libs.PretrainedModels import PretrainedModelsCreator, CCAlexNet

                                                        ## di default carico un modello da mettere nella repo per test antonio
def init(creator: PretrainedModelsCreator, path_mdl: str='AlexNet_2dst__lr=0.0003-40.pth'):
    #print('Instantiating ' )
    #print(type(creator))
    creator.initialize_model(output_class=3)
    #print parameters to be sure that are different model
    #creator.get_info()
    #creator.get_parameters()
    #print("\n\n\n\n")
    creator.load_model(path=path_mdl)
    #creator.get_parameters()

    return creator.return_model()

my_model = init(creator=CCAlexNet())
# print(my_model, type(my_model))
my_model.eval()
# alexnet = models.alexnet(pretrained=True).eval()

raw_labels = str({0: 'empty',1: 'half',2: 'full'})
labels = [i.split(',')[0] for i in list( eval(raw_labels).values() )]

def inference(data):
    image = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])(Image.fromarray(data.astype('uint8'), 'RGB')).unsqueeze(0)
    prediction = torch.nn.functional.softmax(my_model(image)[0], dim=0)

    return dict(zip(labels, map(float, prediction)))

gr.Interface(
            fn=inference,
            inputs=gr.inputs.Image(),
            outputs=gr.outputs.Label(num_top_classes=3)
            ).launch(share=True) #, debug=True Use in Colab
