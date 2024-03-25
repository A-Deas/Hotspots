from PIL import Image

image_set = 'Error Histograms'
training_years = 4

def get_image(image_set, dataset, year):
    if image_set == 'Error Histograms':
        img_path = f'Images/ToLD/{dataset}/Trained on {training_years}.png'
    elif image_set == 'Accuracy Maps':
        img_path = f'Images/{image_set}/{dataset}/{year} {dataset} Accuracy Map.png'
    elif image_set == 'CI Maps':
        img_path = f'Images/{image_set}/{dataset}/{year} {dataset} 95% CI Map.png'
    elif image_set == 'Heat Maps':
        img_path = f'Images/{image_set}/{dataset}/{year} {dataset} Heat Map.png'
    elif image_set == 'Hotspot Maps':
        img_path = f'Images/{image_set}/{dataset}/{year} {dataset} Hotspot Map.png'
    elif image_set == 'Hotspot Accuracy Maps':
        img_path = f'Images/{image_set}/{dataset}/{year} {dataset} Hotspot Accuracy Map.png'
    img = Image.open(img_path)
    return img

def crop_image(img):
    if image_set == 'Error Histograms':
        crop_rectangle = (105, 200, img.width - 200, img.height - 45)
    elif image_set == 'Accuracy Maps':
        crop_rectangle = (170, 190, img.width - 550, img.height - 30)
    elif image_set == 'CI Maps':
        crop_rectangle = (170, 190, img.width - 465, img.height - 30)
    elif image_set == 'Heat Maps':
        crop_rectangle = (170, 205, img.width - 800, img.height - 30)
    elif image_set == 'Hotspot Maps':
        crop_rectangle = (170, 190, img.width - 500, img.height - 25)
    elif image_set == 'Hotspot Accuracy Maps':
        crop_rectangle = (170, 190, img.width - 450, img.height - 25)
    cropped_img = img.crop(crop_rectangle)
    return cropped_img

def save_cropped_image(cropped_img, image_set, dataset, year):
    if image_set == 'Error Histograms':
        output_img_path = f'/Users/deas/Documents/Research/Paper 1/Cropped images for paper/{dataset}/Trained on {training_years}.png'
    elif image_set == 'Accuracy Maps':
        output_img_path = f'/Users/deas/Documents/Research/Paper 1/Cropped images for paper/{dataset}/{year} {dataset} Accuracy Map.png'
    elif image_set == 'CI Maps':
        output_img_path = f'/Users/deas/Documents/Research/Paper 1/Cropped images for paper/{dataset}/{year} {dataset} 95% CI Map.png' 
    elif image_set == 'Heat Maps':
        output_img_path = f'/Users/deas/Documents/Research/Paper 1/Cropped images for paper/{dataset}/{year} {dataset} Heat Map.png' 
    elif image_set == 'Hotspot Maps':
        output_img_path = f'/Users/deas/Documents/Research/Paper 1/Cropped images for paper/{dataset}/{year} {dataset} Hotspot Map.png'
    elif image_set == 'Hotspot Accuracy Maps':
        output_img_path = f'/Users/deas/Documents/Research/Paper 1/Cropped images for paper/{dataset}/{year} {dataset} Hotspot Accuracy Map.png' 
    cropped_img.save(output_img_path)

def main():
    for dataset in ['OD','DR','SVI Disability']:
        for year in range(2020, 2021):
            img = get_image(image_set, dataset, year)
            cropped_img = crop_image(img)
            save_cropped_image(cropped_img, image_set, dataset, year)

if __name__ == "__main__":
    main()