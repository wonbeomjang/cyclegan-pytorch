import os
from config.config import get_config
import shutil

def main():
    config = get_config()


    if not os.path.exists(config.dataset):
        print(f"[!] No dataset in {config.dataset}")
        return

    for image_transfer in os.listdir(config.dataset):
        print(image_transfer)
        if not '2' in image_transfer:
            continue
        from_style, to_style = image_transfer.split('2')
        if not os.path.exists(os.path.join(config.dataset, from_style)):
            a_num = 1000000
            os.makedirs(os.path.join(config.dataset, from_style))
        else:
            a_num = int('1' + str(os.listdir(os.path.join(config.dataset, from_style))[-1].split('.')[0]))

        if not os.path.exists((os.path.join(config.dataset, to_style))):
            b_num = 1000000
            os.makedirs(os.path.join(config.dataset, to_style))
        else:
            b_num = int('1' + str(os.listdir(os.path.join(config.dataset, from_style))[-1].split('.')[0]))
        num = 0

        for image_dir in os.listdir(os.path.join(config.dataset, image_transfer)):
            _from = os.path.join(config.dataset, image_transfer, image_dir)
            if image_dir.endswith('A'):
                num = a_num
                _to = os.path.join(config.dataset, from_style)
            elif image_dir.endswith('B'):
                num = b_num
                _to = os.path.join(config.dataset, to_style)
            else:
                print("[!] unknown dataset format")
                return
            step = 0
            total_step = len(os.listdir(_from))

            for image in os.listdir(_from):
                to_name = str(num)[1:] + '.' + str(image).split(".")[1]
                num += 1
                step += 1
                shutil.move(os.path.join(_from, image), os.path.join(_to, to_name))
                if step % 100 == 0:
                    print(f"[*] [{step}/{total_step}] move image from {_from} to {_to}")

            if image_dir.endswith('A'):
                a_num = num
            elif image_dir.endswith('B'):
                b_num = num

        shutil.rmtree(os.path.join(config.dataset, image_transfer))


if __name__ == '__main__':
    main()
