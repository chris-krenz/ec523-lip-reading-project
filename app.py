import cv2
import torch
import numpy as np
from res_tcn import ResTCN

device = torch.device('cpu')

# loading model 
model = ResTCN(num_classes=500).to(device)
pretrained_model_path = '/Users/ussie/Desktop/EC523/FinalProject/ec523-lip-reading-project/cnn2_1714164171.pth'
pretrained_model_state = torch.load(pretrained_model_path, map_location=device)
model.load_state_dict(pretrained_model_state['model_state_dict'])
model.eval()

# function to preprocess video frames
def preprocess_frame(frame):
    # convert to greyscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resizing to match the input size for the model
    resized_frame = cv2.resize(gray_frame, (112, 112))
    # normalizing pixel values to range [0, 1]
    normalized_frame = resized_frame / 255.0
    # convert frame to a PyTorch tensor
    tensor_frame = torch.FloatTensor(normalized_frame).unsqueeze(0).unsqueeze(0)
    return tensor_frame

# create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

# Threshold for mouth movement detection
movement_threshold = 50

video_capture = cv2.VideoCapture(0)

class_index_to_word = {
    0: 'ABOUT',
    1: 'ABSOLUTELY',
    2: 'ABUSE',
    3: 'ACCESS',
    4: 'ACCORDING',
    5: 'ACCUSED',
    6: 'ACROSS',
    7: 'ACTION',
    8: 'ACTUALLY',
    9: 'AFFAIRS',
    10: 'AFFECTED',
    11: 'AFRICA',
    12: 'AFTER',
    13: 'AFTERNOON',
    14: 'AGAIN',
    15: 'AGAINST',
    16: 'AGREE',
    17: 'AGREEMENT',
    18: 'AHEAD',
    19: 'ALLEGATIONS',
    20: 'ALLOW',
    21: 'ALLOWED',
    22: 'ALMOST',
    23: 'ALREADY',
    24: 'ALWAYS',
    25: 'AMERICA',
    26: 'AMERICAN',
    27: 'AMONG',
    28: 'AMOUNT',
    29: 'ANNOUNCED',
    30: 'ANOTHER',
    31: 'ANSWER',
    32: 'ANYTHING',
    33: 'AREAS',
    34: 'AROUND',
    35: 'ARRESTED',
    36: 'ASKED',
    37: 'ASKING',
    38: 'ATTACK',
    39: 'ATTACKS',
    40: 'AUTHORITIES',
    41: 'BANKS',
    42: 'BECAUSE',
    43: 'BECOME',
    44: 'BEFORE',
    45: 'BEHIND',
    46: 'BEING',
    47: 'BELIEVE',
    48: 'BENEFIT',
    49: 'BENEFITS',
    50: 'BETTER',
    51: 'BETWEEN',
    52: 'BIGGEST',
    53: 'BILLION',
    54: 'BLACK',
    55: 'BORDER',
    56: 'BRING',
    57: 'BRITAIN',
    58: 'BRITISH',
    59: 'BROUGHT',
    60: 'BUDGET',
    61: 'BUILD',
    62: 'BUILDING',
    63: 'BUSINESS',
    64: 'BUSINESSES',
    65: 'CALLED',
    66: 'CAMERON',
    67: 'CAMPAIGN',
    68: 'CANCER',
    69: 'CANNOT',
    70: 'CAPITAL',
    71: 'CASES',
    72: 'CENTRAL',
    73: 'CERTAINLY',
    74: 'CHALLENGE',
    75: 'CHANCE',
    76: 'CHANGE',
    77: 'CHANGES',
    78: 'CHARGE',
    79: 'CHARGES',
    80: 'CHIEF',
    81: 'CHILD',
    82: 'CHILDREN',
    83: 'CHINA',
    84: 'CLAIMS',
    85: 'CLEAR',
    86: 'CLOSE',
    87: 'CLOUD',
    88: 'COMES',
    89: 'COMING',
    90: 'COMMUNITY',
    91: 'COMPANIES',
    92: 'COMPANY',
    93: 'CONCERNS',
    94: 'CONFERENCE',
    95: 'CONFLICT',
    96: 'CONSERVATIVE',
    97: 'CONTINUE',
    98: 'CONTROL',
    99: 'COULD',
    100: 'COUNCIL',
    101: 'COUNTRIES',
    102: 'COUNTRY',
    103: 'COUPLE',
    104: 'COURSE',
    105: 'COURT',
    106: 'CRIME',
    107: 'CRISIS',
    108: 'CURRENT',
    109: 'CUSTOMERS',
    110: 'DAVID',
    111: 'DEATH',
    112: 'DEBATE',
    113: 'DECIDED',
    114: 'DECISION',
    115: 'DEFICIT',
    116: 'DEGREES',
    117: 'DESCRIBED',
    118: 'DESPITE',
    119: 'DETAILS',
    120: 'DIFFERENCE',
    121: 'DIFFERENT',
    122: 'DIFFICULT',
    123: 'DOING',
    124: 'DURING',
    125: 'EARLY',
    126: 'EASTERN',
    127: 'ECONOMIC',
    128: 'ECONOMY',
    129: 'EDITOR',
    130: 'EDUCATION',
    131: 'ELECTION',
    132: 'EMERGENCY',
    133: 'ENERGY',
    134: 'ENGLAND',
    135: 'ENOUGH',
    136: 'EUROPE',
    137: 'EUROPEAN',
    138: 'EVENING',
    139: 'EVENTS',
    140: 'EVERY',
    141: 'EVERYBODY',
    142: 'EVERYONE',
    143: 'EVERYTHING',
    144: 'EVIDENCE',
    145: 'EXACTLY',
    146: 'EXAMPLE',
    147: 'EXPECT',
    148: 'EXPECTED',
    149: 'EXTRA',
    150: 'FACING',
    151: 'FAMILIES',
    152: 'FAMILY',
    153: 'FIGHT',
    154: 'FIGHTING',
    155: 'FIGURES',
    156: 'FINAL',
    157: 'FINANCIAL',
    158: 'FIRST',
    159: 'FOCUS',
    160: 'FOLLOWING',
    161: 'FOOTBALL',
    162: 'FORCE',
    163: 'FORCES',
    164: 'FOREIGN',
    165: 'FORMER',
    166: 'FORWARD',
    167: 'FOUND',
    168: 'FRANCE',
    169: 'FRENCH',
    170: 'FRIDAY',
    171: 'FRONT',
    172: 'FURTHER',
    173: 'FUTURE',
    174: 'GAMES',
    175: 'GENERAL',
    176: 'GEORGE',
    177: 'GERMANY',
    178: 'GETTING',
    179: 'GIVEN',
    180: 'GIVING',
    181: 'GLOBAL',
    182: 'GOING',
    183: 'GOVERNMENT',
    184: 'GREAT',
    185: 'GREECE',
    186: 'GROUND',
    187: 'GROUP',
    188: 'GROWING',
    189: 'GROWTH',
    190: 'GUILTY',
    191: 'HAPPEN',
    192: 'HAPPENED',
    193: 'HAPPENING',
    194: 'HAVING',
    195: 'HEALTH',
    196: 'HEARD',
    197: 'HEART',
    198: 'HEAVY',
    199: 'HIGHER',
    200: 'HISTORY',
    201: 'HOMES',
    202: 'HOSPITAL',
    203: 'HOURS',
    204: 'HOUSE',
    205: 'HOUSING',
    206: 'HUMAN',
    207: 'HUNDREDS',
    208: 'IMMIGRATION',
    209: 'IMPACT',
    210: 'IMPORTANT',
    211: 'INCREASE',
    212: 'INDEPENDENT',
    213: 'INDUSTRY',
    214: 'INFLATION',
    215: 'INFORMATION',
    216: 'INQUIRY',
    217: 'INSIDE',
    218: 'INTEREST',
    219: 'INVESTMENT',
    220: 'INVOLVED',
    221: 'IRELAND',
    222: 'ISLAMIC',
    223: 'ISSUE',
    224: 'ISSUES',
    225: 'ITSELF',
    226: 'JAMES',
    227: 'JUDGE',
    228: 'JUSTICE',
    229: 'KILLED',
    230: 'KNOWN',
    231: 'LABOUR',
    232: 'LARGE',
    233: 'LATER',
    234: 'LATEST',
    235: 'LEADER',
    236: 'LEADERS',
    237: 'LEADERSHIP',
    238: 'LEAST',
    239: 'LEAVE',
    240: 'LEGAL',
    241: 'LEVEL',
    242: 'LEVELS',
    243: 'LIKELY',
    244: 'LITTLE',
    245: 'LIVES',
    246: 'LIVING',
    247: 'LOCAL',
    248: 'LONDON',
    249: 'LONGER',
    250: 'LOOKING',
    251: 'MAJOR',
    252: 'MAJORITY',
    253: 'MAKES',
    254: 'MAKING',
    255: 'MANCHESTER',
    256: 'MARKET',
    257: 'MASSIVE',
    258: 'MATTER',
    259: 'MAYBE',
    260: 'MEANS',
    261: 'MEASURES',
    262: 'MEDIA',
    263: 'MEDICAL',
    264: 'MEETING',
    265: 'MEMBER',
    266: 'MEMBERS',
    267: 'MESSAGE',
    268: 'MIDDLE',
    269: 'MIGHT',
    270: 'MIGRANTS',
    271: 'MILITARY',
    272: 'MILLION',
    273: 'MILLIONS',
    274: 'MINISTER',
    275: 'MINISTERS',
    276: 'MINUTES',
    277: 'MISSING',
    278: 'MOMENT',
    279: 'MONEY',
    280: 'MONTH',
    281: 'MONTHS',
    282: 'MORNING',
    283: 'MOVING',
    284: 'MURDER',
    285: 'NATIONAL',
    286: 'NEEDS',
    287: 'NEVER',
    288: 'NIGHT',
    289: 'NORTH',
    290: 'NORTHERN',
    291: 'NOTHING',
    292: 'NUMBER',
    293: 'NUMBERS',
    294: 'OBAMA',
    295: 'OFFICE',
    296: 'OFFICERS',
    297: 'OFFICIALS',
    298: 'OFTEN',
    299: 'OPERATION',
    300: 'OPPOSITION',
    301: 'ORDER',
    302: 'OTHER',
    303: 'OTHERS',
    304: 'OUTSIDE',
    305: 'PARENTS',
    306: 'PARLIAMENT',
    307: 'PARTIES',
    308: 'PARTS',
    309: 'PARTY',
    310: 'PATIENTS',
    311: 'PAYING',
    312: 'PEOPLE',
    313: 'PERHAPS',
    314: 'PERIOD',
    315: 'PERSON',
    316: 'PERSONAL',
    317: 'PHONE',
    318: 'PLACE',
    319: 'PLACES',
    320: 'PLANS',
    321: 'POINT',
    322: 'POLICE',
    323: 'POLICY',
    324: 'POLITICAL',
    325: 'POLITICIANS',
    326: 'POLITICS',
    327: 'POSITION',
    328: 'POSSIBLE',
    329: 'POTENTIAL',
    330: 'POWER',
    331: 'POWERS',
    332: 'PRESIDENT',
    333: 'PRESS',
    334: 'PRESSURE',
    335: 'PRETTY',
    336: 'PRICE',
    337: 'PRICES',
    338: 'PRIME',
    339: 'PRISON',
    340: 'PRIVATE',
    341: 'PROBABLY',
    342: 'PROBLEM',
    343: 'PROBLEMS',
    344: 'PROCESS',
    345: 'PROTECT',
    346: 'PROVIDE',
    347: 'PUBLIC',
    348: 'QUESTION',
    349: 'QUESTIONS',
    350: 'QUITE',
    351: 'RATES',
    352: 'RATHER',
    353: 'REALLY',
    354: 'REASON',
    355: 'RECENT',
    356: 'RECORD',
    357: 'REFERENDUM',
    358: 'REMEMBER',
    359: 'REPORT',
    360: 'REPORTS',
    361: 'RESPONSE',
    362: 'RESULT',
    363: 'RETURN',
    364: 'RIGHT',
    365: 'RIGHTS',
    366: 'RULES',
    367: 'RUNNING',
    368: 'RUSSIA',
    369: 'RUSSIAN',
    370: 'SAYING',
    371: 'SCHOOL',
    372: 'SCHOOLS',
    373: 'SCOTLAND',
    374: 'SCOTTISH',
    375: 'SECOND',
    376: 'SECRETARY',
    377: 'SECTOR',
    378: 'SECURITY',
    379: 'SEEMS',
    380: 'SENIOR',
    381: 'SENSE',
    382: 'SERIES',
    383: 'SERIOUS',
    384: 'SERVICE',
    385: 'SERVICES',
    386: 'SEVEN',
    387: 'SEVERAL',
    388: 'SHORT',
    389: 'SHOULD',
    390: 'SIDES',
    391: 'SIGNIFICANT',
    392: 'SIMPLY',
    393: 'SINCE',
    394: 'SINGLE',
    395: 'SITUATION',
    396: 'SMALL',
    397: 'SOCIAL',
    398: 'SOCIETY',
    399: 'SOMEONE',
    400: 'SOMETHING',
    401: 'SOUTH',
    402: 'SOUTHERN',
    403: 'SPEAKING',
    404: 'SPECIAL',
    405: 'SPEECH',
    406: 'SPEND',
    407: 'SPENDING',
    408: 'SPENT',
    409: 'STAFF',
    410: 'STAGE',
    411: 'STAND',
    412: 'START',
    413: 'STARTED',
    414: 'STATE',
    415: 'STATEMENT',
    416: 'STATES',
    417: 'STILL',
    418: 'STORY',
    419: 'STREET',
    420: 'STRONG',
    421: 'SUNDAY',
    422: 'SUNSHINE',
    423: 'SUPPORT',
    424: 'SYRIA',
    425: 'SYRIAN',
    426: 'SYSTEM',
    427: 'TAKEN',
    428: 'TAKING',
    429: 'TALKING',
    430: 'TALKS',
    431: 'TEMPERATURES',
    432: 'TERMS',
    433: 'THEIR',
    434: 'THEMSELVES',
    435: 'THERE',
    436: 'THESE',
    437: 'THING',
    438: 'THINGS',
    439: 'THINK',
    440: 'THIRD',
    441: 'THOSE',
    442: 'THOUGHT',
    443: 'THOUSANDS',
    444: 'THREAT',
    445: 'THREE',
    446: 'THROUGH',
    447: 'TIMES',
    448: 'TODAY',
    449: 'TOGETHER',
    450: 'TOMORROW',
    451: 'TONIGHT',
    452: 'TOWARDS',
    453: 'TRADE',
    454: 'TRIAL',
    455: 'TRUST',
    456: 'TRYING',
    457: 'UNDER',
    458: 'UNDERSTAND',
    459: 'UNION',
    460: 'UNITED',
    461: 'UNTIL',
    462: 'USING',
    463: 'VICTIMS',
    464: 'VIOLENCE',
    465: 'VOTERS',
    466: 'WAITING',
    467: 'WALES',
    468: 'WANTED',
    469: 'WANTS',
    470: 'WARNING',
    471: 'WATCHING',
    472: 'WATER',
    473: 'WEAPONS',
    474: 'WEATHER',
    475: 'WEEKEND',
    476: 'WEEKS',
    477: 'WELCOME',
    478: 'WELFARE',
    479: 'WESTERN',
    480: 'WESTMINSTER',
    481: 'WHERE',
    482: 'WHETHER',
    483: 'WHICH',
    484: 'WHILE',
    485: 'WHOLE',
    486: 'WINDS',
    487: 'WITHIN',
    488: 'WITHOUT',
    489: 'WOMEN',
    490: 'WORDS',
    491: 'WORKERS',
    492: 'WORKING',
    493: 'WORLD',
    494: 'WORST',
    495: 'WOULD',
    496: 'WRONG',
    497: 'YEARS',
    498: 'YESTERDAY',
    499: 'YOUNG',
}



'''

# capture frames from the video stream
while True:
    
    ret, frame = video_capture.read()

    # preprocess the frame
    tensor_frame = preprocess_frame(frame)

    # make prediction
    with torch.no_grad():
        # forward pass through the model
        prediction = model(tensor_frame)

        # get predicted word index
        predicted_word_index = torch.argmax(prediction).item()
        
        # get preidicted word based on index using the dictionary 
        predicted_word = class_index_to_word.get(predicted_word_index, 'Unknown')

        # display the prediction
        cv2.putText(frame, "Predicted word: {}".format(predicted_word), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # display frame
    cv2.imshow('Video', frame)

    # 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

#  capture frames from the video stream
while True:
    # capture frames 
    ret, frame = video_capture.read()

    # apply background subtraction to isolate moving objects
    fg_mask = bg_subtractor.apply(frame)

    # enhance detection
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, (5, 5))

    # find contours 
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # iterate over contours
    for contour in contours:
        # calculate area of contour
        area = cv2.contourArea(contour)

        # filter out small contours
        if area > 1000:

            # get bounding box of contour
            x, y, w, h = cv2.boundingRect(contour)

            # extract mouth region
            mouth_roi = frame[y:y+h, x:x+w]

            # preprocess the frame
            tensor_frame = preprocess_frame(mouth_roi)

            # calculate mean intensity of the mouth region
            mean_intensity = np.mean(mouth_roi)

            # doing this so that model predicts only when mouth moves 
            # make prediction if the mean intensity exceeds the threshold
            if mean_intensity > movement_threshold:
                with torch.no_grad():
                    # forward pass through the model
                    prediction = model(tensor_frame)

                    # get predicted word index
                    predicted_word_index = torch.argmax(prediction).item()
                    
                    # get predicted word based on index using the dictionary 
                    predicted_word = class_index_to_word.get(predicted_word_index)

                    # display the prediction
                    cv2.putText(frame, "Predicted word: {}".format(predicted_word), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # display frame
    cv2.imshow('Video', frame)

    # 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close app
video_capture.release()
cv2.destroyAllWindows()
