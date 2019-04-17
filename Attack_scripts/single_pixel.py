from util import *
import torch.nn.functional as F
from differential_evolution import differential_evolution
# from scipy.optimize import differential_evolution
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x) / 5)

        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = (r / 255.0 - 0.4914) / 0.2023
            imgs[count, 1, x_pos, y_pos] = (g / 255.0 - 0.4822) / 0.1994
            imgs[count, 2, x_pos, y_pos] = (b / 255.0 - 0.4465) / 0.2010
        count += 1

    return imgs


def predict_classes(xs, img, target_calss, net, minimize=True):
    imgs_perturbed = perturb_image(xs, img.clone())
    input = Variable(imgs_perturbed).to(device)
    predictions = F.softmax(net(input)).data.cpu().numpy()[:, target_calss]

    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_class, net, targeted_attack=False, verbose=False):
    attack_image = perturb_image(x, img.clone())
    input = Variable(attack_image).to(device)
    confidence = F.softmax(net(input)).data.cpu().numpy()[0]
    predicted_class = np.argmax(confidence)

    if (verbose):
        print("Confidence: %.4f" % confidence[target_class])
    if (targeted_attack and predicted_class == target_class) or (
            not targeted_attack and predicted_class != target_class):
        return True


def attack_max_iter(img_size, img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
    # img: 1*3*W*H tensor
    # label: a number

    if (verbose):
        print("\nStart to attack image")

    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    bounds = [(0, img_size), (0, img_size), (0, 255), (0, 255), (0, 255)] * pixels

    popmul = max(1, popsize / len(bounds))

    predict_fn = lambda xs: predict_classes(
        xs, img, target_class, net, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_class, net, targeted_attack, verbose)

    inits = np.zeros([int(popmul * len(bounds)), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i * 5 + 0] = np.random.random() * img_size
            init[i * 5 + 1] = np.random.random() * img_size
            init[i * 5 + 2] = np.random.normal(128, 127)
            init[i * 5 + 3] = np.random.normal(128, 127)
            init[i * 5 + 4] = np.random.normal(128, 127)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul, mutation=(0.5, 1),
                                           recombination=0.7, atol=-1, callback=callback_fn, polish=True, init=inits)

    attack_image = perturb_image(attack_result.x, img)
    return attack_image