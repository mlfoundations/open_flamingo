from functools import partial
from typing import Dict, Any, Tuple

import torch
from open_flamingo.eval.classification import compute_per_sample_probs, \
    compute_per_sample_loss
from open_flamingo.eval.evaluation_utils import FlamingoModelLoader, \
    get_context_images, get_context_text, prepare_batch_images

# classnames via https://github.com/mlfoundations/wise-ft/blob/master/src/datasets/imagenet_classnames.py#L1
openai_imagenet_classnames = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch",
    "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee",
    "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl",
    "fire salamander",
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog",
    "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle",
    "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard",
    "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon",
    "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake",
    "garter snake", "water snake",
    "vine snake", "night snake", "boa constrictor", "African rock python",
    "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper",
    "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion",
    "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow",
    "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse",
    "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw",
    "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan",
    "duck",
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna",
    "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm",
    "nematode", "conch",
    "snail", "slug", "sea slug", "chiton", "chambered nautilus",
    "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster",
    "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo",
    "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule",
    "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher",
    "oystercatcher",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale",
    "dugong", "sea lion",
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu",
    "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound",
    "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound",
    "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound",
    "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki",
    "Scottish Deerhound",
    "Weimaraner", "Staffordshire Bull Terrier",
    "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier",
    "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier",
    "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier",
    "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier",
    "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier",
    "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever",
    "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever",
    "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog",
    "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel",
    "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog",
    "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog",
    "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler",
    "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff",
    "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute",
    "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger",
    "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond",
    "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle",
    "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf",
    "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog",
    "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat",
    "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar",
    "lion", "tiger",
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear",
    "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle",
    "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant",
    "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada",
    "leafhopper",
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly",
    "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly",
    "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare",
    "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig",
    "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox",
    "water buffalo", "bison",
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest",
    "impala (antelope)",
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo",
    "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey",
    "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset",
    "white-headed capuchin",
    "howler monkey", "titi monkey", "Geoffroy's spider monkey",
    "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant",
    "red panda",
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish",
    "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya",
    "academic gown",
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship",
    "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can",
    "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen",
    "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn",
    "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon",
    "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker",
    "military hat (bearskin or shako)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle",
    "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh",
    "bolo tie", "poke bonnet",
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie",
    "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle",
    "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle",
    "cannon", "canoe",
    "can opener", "cardigan", "car mirror", "carousel", "tool kit",
    "cardboard box / carton",
    "car wheel", "automated teller machine", "cassette", "cassette player",
    "castle", "catamaran",
    "CD player", "cello", "mobile phone", "chain", "chain-link fence",
    "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet",
    "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs",
    "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock",
    "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet",
    "cowboy boot",
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate",
    "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk",
    "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch",
    "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome",
    "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan",
    "electric guitar",
    "electric locomotive", "entertainment center", "envelope",
    "espresso machine", "face powder",
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen",
    "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen",
    "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball",
    "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille",
    "grocery store", "guillotine",
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer",
    "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester",
    "hatchet",
    "holster", "home theater", "honeycomb", "hook", "hoop skirt",
    "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron",
    "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad",
    "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap",
    "letter opener", "library",
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick",
    "slip-on shoe", "lotion",
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass",
    "messenger bag",
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca",
    "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet",
    "megalith", "microphone",
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt",
    "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem",
    "monastery", "monitor",
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net",
    "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van",
    "muzzle", "metal nail",
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk",
    "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt",
    "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel",
    "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute",
    "parallel bars", "park bench",
    "parking meter", "railroad car", "patio", "payphone", "pedestal",
    "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum",
    "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle",
    "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium",
    "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole",
    "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill",
    "prayer rug",
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag",
    "purse", "quill",
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope",
    "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera",
    "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair",
    "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin",
    "salt shaker", "sandal",
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus",
    "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine",
    "shield", "shoe store",
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel",
    "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule",
    "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser",
    "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar",
    "space heater",
    "space shuttle", "spatula", "motorboat", "spider web", "spindle",
    "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum",
    "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher",
    "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen",
    "suspension bridge",
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch",
    "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television",
    "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne",
    "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole",
    "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle",
    "trimaran", "tripod",
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile",
    "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase",
    "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin",
    "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft",
    "sink",
    "washing machine", "water bottle", "water jug", "water tower",
    "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle",
    "airplane wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat",
    "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light",
    "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream",
    "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage",
    "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash",
    "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple",
    "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit",
    "cherimoya (custard apple)", "pomegranate",
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza",
    "pot pie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff",
    "coral reef",
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley",
    "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper",
    "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra",
    "stinkhorn mushroom",
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob",
    "toilet paper"
]
# Maps numeric class ids to labels
IMAGENET_1K_CLASS_ID_TO_LABEL = dict(zip(range(len(openai_imagenet_classnames)),
                                         openai_imagenet_classnames))


def compute_per_sample_probs_and_loss(
        imagenet_class_name: str,
        context_text: str,
        context_ids: torch.Tensor,
        _imagenet_prompt, eoc_token, eoc_token_id,
        batch_size, tokenizer, tokenizer_kwargs, device, context_len, model,
        context_precomputed) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_text = [context_text
                  + _imagenet_prompt(imagenet_class_name, False)
                  + eoc_token] * batch_size

    full_batch_encodings = tokenizer(batch_text, **tokenizer_kwargs)

    # full_batch_input_ids has shape [batch_size, seq_len], but we
    # only need to run inference on the [batch_size,
    # context_len:] inputs that have not been precomputed and
    # vary per class.
    full_batch_input_ids = full_batch_encodings["input_ids"].to(device)
    full_batch_attention_mask = full_batch_encodings[
        "attention_mask"].to(device)

    # Sanity check that the encoded inputs with context are the same
    # as the encoded context alone, for every example in the batch
    assert torch.all(context_ids[0, :] == full_batch_input_ids[:,
                                          :context_len]).item()

    # Clone the nested structure of the past key values
    past_key_values = tuple(
        [tuple([x.clone() for x in inner]) for inner in
         context_precomputed.past_key_values])

    # Compute the outputs without recomputing context representations.
    outputs = model(
        vision_x=None,
        lang_x=full_batch_input_ids[:, context_len:],
        attention_mask=full_batch_attention_mask,
        use_cached_vision_x=True,
        clear_conditioned_layers=False,
        past_key_values=past_key_values,
        use_cache=True)

    logits = torch.concat(
        (context_precomputed.logits, outputs.logits), 1)

    per_sample_probs = compute_per_sample_probs(
        encodings=full_batch_encodings,
        tokenizer=tokenizer,
        logits=logits,
        eoc_token_id=eoc_token_id)
    per_sample_loss = compute_per_sample_loss(
        encodings=full_batch_encodings,
        tokenizer=tokenizer,
        logits=logits,
        eoc_token_id=eoc_token_id)
    return per_sample_probs, per_sample_loss


def _imagenet_prompt(class_name, eos_token: str, is_context: bool = True):
    """Construct an imagenet prompt for a given label."""
    prefix = "<image>A photo of a "
    if is_context:
        return prefix + class_name.strip()
    else:
        # Not a context example; insert EOS token before the class name
        # so that we can compute the loss on the class name tokens only.
        return prefix + eos_token + class_name.strip()


def get_imagenet_prompt(x: dict, eos_token: str,
                        is_context: bool = True) -> str:
    """Construct an ImageNet prompt for an example, using its label."""
    return _imagenet_prompt(x['class_name'], is_context=is_context,
                            eos_token=eos_token)


def infer(rank, queue, flamingo_loader: FlamingoModelLoader,
          batch, in_context_samples,
          batch_size: int, tokenizer_kwargs: Dict[str, Any],
          num_shots: int, effective_num_shots: int):
    """Each subprocess will run this function on a different
    GPU which is indicated by the parameter `rank`."""
    device = torch.device(f"cuda:{rank}")
    print(f'loading model on device {rank}...')
    model, image_processor, tokenizer = flamingo_loader.load(rank)
    model.to(device)
    model.eval()
    print(f'finished loading model on device {rank}.')

    eoc_token = "<|endofchunk|>"
    eoc_token_id = tokenizer.additional_special_tokens_ids[
        tokenizer.additional_special_tokens.index(eoc_token)
    ]

    print(f'processing context images and text on device {rank}.')
    context_images = get_context_images(image_processor=image_processor,
                                        in_context_samples=in_context_samples,
                                        num_shots=num_shots)

    _get_imagenet_prompt = partial(get_imagenet_prompt,
                                   eos_token=tokenizer.eos_token)
    context_text = get_context_text(_get_imagenet_prompt,
                                    in_context_samples=in_context_samples,
                                    effective_num_shots=effective_num_shots,
                                    num_shots=num_shots)

    batch_images = prepare_batch_images(batch=batch,
                                        image_processor=image_processor,
                                        context_images=context_images,
                                        num_shots=num_shots)
    # Process the images only once.
    batch_images = batch_images.to(device)
    model._process_media(vision_x=batch_images)

    # Process the context text only once.
    context_encodings = tokenizer([context_text] * batch_size,
                                  **tokenizer_kwargs)
    context_ids = context_encodings['input_ids'].to(device)
    context_len = context_ids.shape[-1]
    context_precomputed = model(None, context_ids,
                                use_cached_vision_x=True,
                                clear_conditioned_layers=False,
                                use_cache=True)
    print(f'finished processing context images and text on device {rank}.')

    # Padding from right allows efficient precomputing of context activations.
    tokenizer.padding_side = "right"

    while True:

        item = queue.get()

        if item is None:  # check for sentinel value
            break

        else:
            imagenet_class_id, imagenet_class_name, return_dict = item
            print(
                f"got class {imagenet_class_name} ({imagenet_class_id}) "
                f"on process {rank}; running eval...")
            per_sample_probs, _ = \
                compute_per_sample_probs_and_loss(
                    imagenet_class_name, context_text, context_ids,
                    _imagenet_prompt, eoc_token,
                    eoc_token_id, batch_size,
                    tokenizer, tokenizer_kwargs, device, context_len,
                    model, context_precomputed)
            print(f"successfully computed per sample probs on device {rank} "
                  f"for class {imagenet_class_name}.")
            per_sample_probs = per_sample_probs.detach().cpu()
            return_dict[imagenet_class_id] = per_sample_probs
