# import random
# from moviepy.editor import vfx

# def apply_instagram_theme(clip, intensity="medium"):

#     # 🎨 Filters
#     filters = [
#         lambda c: c.fx(vfx.colorx, 1.2),
#         lambda c: c.fx(vfx.lum_contrast, 10, 50, 128),
#         lambda c: c.fx(vfx.blackwhite),
#         lambda c: c.fx(vfx.invert_colors),
#     ]

#     # 🎥 Motion effects
#     motions = [
#         lambda c: c.resize(lambda t: 1 + 0.03 * t),
#         lambda c: c.resize(lambda t: 1 - 0.02 * t),
#     ]

#     # 🎬 Fade
#     def add_fade(c):
#         d = min(0.3, c.duration / 4)
#         return c.fadein(d).fadeout(d)

#     # 🎯 Intensity logic
#     if intensity == "high":
#         clip = random.choice(filters)(clip)
#         clip = random.choice(motions)(clip)

#     elif intensity == "medium":
#         if random.random() > 0.5:
#             clip = random.choice(filters)(clip)
#         if random.random() > 0.5:
#             clip = random.choice(motions)(clip)

#     else:
#         if random.random() > 0.7:
#             clip = random.choice(filters)(clip)

#     clip = add_fade(clip)

#     return clip