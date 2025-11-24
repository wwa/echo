import pytesseract
import mss
import mss.tools
from PIL import ImageGrab, Image

#Import(own)
from Toolkit import toolspec, AttrDict
from Toolkit import BaseToolkit

class BaseToolkitVisualPerception(BaseToolkit):
  def __init__(self):
    super().__init__()

  def screenshot(self, title=None):
    self.trace.info("ACTION: Capturing your screen (primary monitor).")
    with mss.mss() as sct:
      monitor = sct.monitors[1]  # 0 = all, 1 = primary
      img = sct.grab(monitor)
      img_pil = Image.frombytes("RGB", img.size, img.rgb)
      self.data.screenshot = img_pil
    return "{status: success}"

  def selectImage(self, image=None):
    if image is None:
      try:
        image = self.data.clipboard
        if not isinstance(image, Image.Image):
          image = Image.open(image)
      except:
        image = None
    if image is None:
      image = self.data.screenshot
    return image

  @toolspec(
    desc="Optical character recognition to extract text from image. Category: input, image",
    args={"image": {"type": "string",
                    "description": "Image file to OCR. If not specified, clipboard or screenshot will be used automatically."}},
    reqs=[]
  )
  def ocr(self, image=None):
    image = self.selectImage(image)
    return f"{{status: success, content:{pytesseract.image_to_string(image)}}}"

  @toolspec(
    desc="""
        Performs image processing using vision model. 
        Clipboard image or screenshot will be used automatically.
        Category: input, image""",
    args={"prompt": {"type": "string",
                     "description": "Prompt for vision model. User prompt will also be available for context."}},
    reqs=["prompt"],
    prompt="Plan: If clipboard data seems short or not suitable, consider calling vision instead."
  )
  def vision(self, prompt, img=None):
    img = self.selectImage(img)
    ocr = self.ocr(img)
    res = self.openai.chat.completions.create(
      model=self.openai_vision_model,
      max_tokens=500,
      messages=[{
        "role": "system",
        "content": f"""
              You are a subordinate function of an assistant called Echo.
              Echo determined that users request is related to this image and called you.
              You are not talking to the user directly. Be succint. Avoid pleasentries, appologizing and trivial explanations.
              OCR data of the image is provided below. 
              For context the user request to Echo was: {{{self.data.prompt}}}
              If user request is about textual data take a guess on what's important, extract it from OCR and return it verbatim.
              If user request is not about text or if OCR data is not useful to the request, proceed as you see fit yourself.
              <ocr>
              {ocr}
              </ocr>
            """
      },
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64(img)}"}
          ]
        }]
    )
    return res.choices[0].message.content