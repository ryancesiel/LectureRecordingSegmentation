from whoosh.index import create_in
from whoosh.fields import *
import os

INDEX_DIR = 'index'

schema = Schema(course_name=TEXT(stored=True), segment_id=ID(stored=True), transcript=TEXT(stored=True))

if not os.path.exists(INDEX_DIR):
    os.mkdir(INDEX_DIR)

ix = create_in(INDEX_DIR, schema)
writer = ix.writer()
writer.add_document(course_name=u"Intro to Psychology Fall 2011", segment_id=u"1",
transcript=u"""The following content is provided under a Creative Commons license. Your support will help MIT OpenCourseWare continue to offer high quality educational resources for free. To make a donation or view additional materials from hundreds of MIT courses, visit MIT OpenCourseWare at ocw.mit.edu.
PROFESSOR: Good afternoon. Congratulations for braving it through what's now become a weekly snow disaster. This week's maybe three of them or something. My name's John Gabrieli. This is Introductory to Psychology, 9.00.
This is a course about you. The entire course is what do we understand in a scientific way about human nature-- how people's minds work, how people's brains work that supports their mind. This entire course is about what's a scientific way to understanding how people feel, think, and act in the world.
And so we're trying to say that we constantly think you must in your everyday life think about why do you have your preferences, your desires? What's easy for you? What's hard for you? What's delightful for you? Why do other people behave the way they do? How do they think? How do they feel?
And so there's a lot of realms of this that are tough to get to by science. But what we're going to focus on this semester is where the scientific approach has shed light in the way that we used to think about experiments and evidence, about how humans tick.
And as we go through this semester, we'll talk about the brain, we'll talk a fair bit about chapters from this book, The Man Who Mistook His Wife for a Hat, from Oliver Sacks. It was a bestseller even when it wasn't [? a ScienceWare ?] course. It's a great book. You'll enjoy it. Short, really fun chapters.
We'll talk about how we perceive the world; how we see; especially, a little bit, how we hear; how we think; how we feel; personality; how we differ from one to the other; and what we're sort of like; and how we behave in the world; development from childhood and infancy through adolescence, through young adulthood, where you are mostly, through getting older, where I am; social interaction, how we behave in groups and think about other people; and variation in the mental health or psychopathology.
And increasingly, we understand that there's a huge number of people who, at some moment in their life or another, struggle with some aspect of mental health. And then we'll focus a lot on, not only the psychological aspects of what we study in terms of behavior, but also the brain basis of that, and think a little bit about to what extent the mind is what the brain does, to what extent the mind is what the brain does.
And so for every dimension of being a human being that we'll talk about, we'll also talk about what we understand currently from the neurological and neuroscientific literature about how the human brain supports and contributes to different aspects of being a person. OK.
So everybody who works in a certain field thinks that their field is really, really, really special, right? So here's why psychology is really, really, really special. So it's really, really special, I think, most of all, because every endeavor that we undertake at a university or in society as a whole-- it's about people, right, except for when we think about the rest of nature. But people study biology, chemistry, and physics. And they think, right, that the sun orbits the earth for some period of time. And then they think it's the other way around currently, right?
OK, so people come up with these conclusions. Even though we're trying to understand nature, it's people who make certain investments in economics or behave in a certain way or vote in a certain way. It's people who make music and appreciate music, make art and appreciate art, read and write literature, right?
So in all these dimensions, there's something very fundamental about what it is about the human mind that gives birth to these areas of inquiry and how those areas, domains of human experience, are enacted.
So my only goal today is to try to convince you in a number of different ways that we're not simple video camera in our minds between our ears, recording the world in some objective, simple way, that even the simplest, most obvious things are interpretations of the world around us at many different levels of thought and feeling and perception.
And then our minds, the way our minds are constructed, determines the world that we experience, that we see, that we act upon. And even very simple things that we think are pretty objective and simple, right in front of our eyes, are determined by inferences and deductions that our mind makes, weighing sources of evidence in the world and coming to conclusions about what's around us, what we hear, what we see, and how we think. """
)

writer.add_document(course_name=u"Intro to Psychology Fall 2011", segment_id=u"2",
transcript=u"""So let's start with seeing. If your vision is reasonable, we say we see something, we believe it, right? So let's start with something very simple-- these lines. So one of the tough things about psychology is ever since the Internet came into existence, people know every cool thing there is to know, right? OK. I can tell you when I began teaching, people said, oh my gosh, I've never seen such a thing. It's unbelievable! And then now, it's like two thirds of the class is like, yeah, I've got that on my computer at home. We did that in third grade or whatever. So all I'm saying is enjoy the ones you haven't seen before, don't ruin it for your neighbors today, because it's harder and harder to surprise the world in a nice way, right?

OK, but let's look at these lines for a moment here. And perhaps you'll have the sense, and maybe-- is it glaring up there, sir? Let's see. OK, is that better? OK. Maybe not.

So you might have the sense that this line is a different length than this line. And this might be somewhere intermediate, right? Now you know, because of psychology, it's all a trick. But what's simpler than the length of a line? What's more objective in some sense than the length of a line? But if we look at the actual lengths, they're all literally identical. But that center part looks different.

So what does it mean for it to look different? It means our minds are determining as simple a thing as how long a line is depending on the other information surrounding it. It's an interpretation in context. If we're simply looking, the lines will look the same.

Let's try another one. It's remarkable that those two lines are identical in length.

[LAUGHTER]

PROFESSOR: OK, all right. It's OK to test the limits of the credibility of the audience, right? All right. Yeah.

Of course, if our visual system were ludicrously off, we'd be constantly walking into walls and falling out windows and things like that, right, if we were misestimating at that length. So the idea where we have visual illusions-- and I'll show you some more that I think you'll be impressed by-- it's not that our visual system is messed up or that psychologists think it's hilarious to trick us.

It's that lots of things our visual system is a brilliant at, but it's brilliant by having certain laws or principles that it follows. And we can show this following those principles by seeing that when we mess with the typical circumstances, those principles calculate the wrong answer.

So here's another one. So, to most people, which line looks bigger, the one in the middle or the one on the side? I know you know it's all a trick, right? OK. What could be more obvious than that this is longer? It's just a simple line, but if we draw red lines on top of it then move them over here, they're dead identical.

The central circle-- does one of them, the middle circle, look larger than the other? Now you already know, intellectually, that it will turn out those two circles in the middle will be the same. But you have to convince yourself that it still looks like they're different. Here there in red. Here they are next to each other. They're identical. Again, this is evidence that, even for a simple thing like the size of a circle, your mind is making inferences. And there are principles and laws that it's following that determine what it is you think that you see.

Here is two monsters chasing each other. But in fact, they're identical in size. The perspective cues make the more distant one look much bigger. This is from Ted Adelson. This is a beautiful demonstration of an illusion. Ted Adelson's in the psychology department. There's a letter A here. And believe it or not, there's a letter B there. Let's see if this looks any better when it goes like this. It doesn't. All right.

So one of the important things about illusions, demonstrations in this class-- and you will learn this as we go along-- is occasionally they fail, and we come back and discover what the lesson of that is. So I'm just telling you it's showing you on my monitor much brighter. It always has before. We'll adjust that. So I'm going to skip this, but I'll show you another time, because it's so good.

And I'm going to feel bad about this. OK. Now, let's see. This'll work. All the same shade of grey, right?

[LAUGHTER]

PROFESSOR: Did that work reasonably from where you sat? We'll try a few more. Maybe. For some reason, my connection's always like this, sorry.

Does that one look lighter than that one that way? Yeah. Now they look radically different, right? It's the same grey constantly. But again, the context is hugely determining how to bright you see that grey. There it is.

Two boxes equal grey. So things as simple as how bright something is or how long something is depend on interpretation. Here's an illusion from Roger Shepard. It's kind of great. So here's two kind of different looking tables, right? But they're not that different. And watch. There goes one tabletop. You're not impressed that those are identical tables? OK. Want me to do it again? That's the identical tabletop. To me, the one on the left looks pretty rectangular and the one on the right looks pretty square-ish. You're not easy to impress, are you?

[LAUGHTER]

PROFESSOR: You see that those two bars are moving together at the same time. Does it look like they're little steps? It'll show you. All right, fine. It's just like that, but now you add those bars. Does it look like little steps?

[LAUGHTER]

PROFESSOR: One more of this kind. This is kind of fun. You see the way that the mask is turning? It always looks like it's towards you, even though I'm-- one of the rotations-- it's because of the way you're interpreting the light is influencing how you interpreting what's-- OK.

So that's simply a consequence, as far as people understand that, that the source of the illumination is not where you're used to, so you're misinterpreting where the illumination is coming from for the depth of the face, what's front and what's back, whether the nose is sticking in or sticking out.

OK. So again, the point in these illusions is, even for very simple things our, minds make certain assumptions about how we interpret the world. And that drives everything that we see and how we act upon what we see.

So at a slightly higher or more conceptual level, I need your help. Now, there's lots of these things we'll do this semester where you get to participate. The fun thing about-- I said this course was about you-- when you could have thought that was a bit rhetorical, it's not. It's truly about you. So you get to be your own laboratory. We get to share a laboratory sitting here. And what I'm going to do is ask for you to participate. You don't have to do any of these things sitting at your seat, but I think it's usually fun to do them.

So what's going to happen is I'm going to show you a drawing. If the people to my left-- so about in the middle, but you can decide for yourself-- about this way, let's have you be Group A if you're willing to be that way. All right. Because of that, I can't call you-- I was going to call you guys Group B, but I already see that's getting me in trouble. So we'll call it Group B, but that really means equals A. But I'll just call it B, OK? So A and B, OK?

So what I need is Group B-- B for best, A for awesome, OK.

[LAUGHTER]

PROFESSOR: --Group B to close your eyes for a moment. Group B, if you want to have fun with this, close your eyes for a moment. Group A, you're gonna see some instructions, and read them silently to yourself. And then I'll ask you a question about the picture. OK, Group A, you're now reading. Group B has your eyes closed. So read the instructions silently to yourself.

OK? Now Group A, close your eyes. Everybody has their eyes closed for a moment. Everybody has their eyes closed. Now Group B, look at your instructions. So A has their eyes closed, B is reading instructions. OK?

Everybody's eyes are open now. Everybody's eyes are open. Here's your picture. Take it in and I'm going to ask you a few questions about it. Look at it for a moment and inspect it. OK, here we go, ready? So just out loud-- was there an automobile in the picture?

AUDIENCE: No.

PROFESSOR: OK. See, this is a smart class. We're gonna have a-- Was there a man in the picture?

AUDIENCE: Yes.

PROFESSOR: Was there a woman in the picture?

AUDIENCE: Yes.

AUDIENCE: No.

PROFESSOR: OK. This side again, woman in the picture?

AUDIENCE: No.

PROFESSOR: All right, all right. OK, a child?

AUDIENCE: No.

PROFESSOR: An animal?

AUDIENCE: No.

AUDIENCE: Yes.

PROFESSOR: Ah. OK. And now it gets a little wild. OK? A whip?

AUDIENCE: Yes.

PROFESSOR: OK. A sword?

AUDIENCE: Yes.

PROFESSOR: All right, a man's hat?

AUDIENCE: Yes.

PROFESSOR: A ball?

AUDIENCE: Yes.

AUDIENCE: No.

PROFESSOR: A fish?

AUDIENCE: Yes.

PROFESSOR: All right, so there's disagreement. And that's-- we're a democracy, right? So all these things are big setups, right? So here's what happened. Group A was told they were gonna look at a picture of a trained seal act. And Group B got the identical instructions, but they were told you're gonna look at a costume ball.

So you had an expectation of what you were going to see. That expectation drives your interpretation of the very thing you see next, which is this picture. OK?

[LAUGHTER]

[CHATTER]

PROFESSOR: OK, is that all right? All right. And this is just for fun, right? It's a set up. You're participating nicely. But in the world, when groups that are arguing with each other about things like peace settlements, read a document, or make a statement, how much do you think the perspective they start with guides the interpretation of what they read or what they hear?

Because you didn't have big stakes in this. You weren't going, I believe in fish and if I don't see a fish, I know things aren't just and my group will be not treated fairly. You're not emotionally invested in, probably, whether there was a fish present. So your interpretation, your beliefs guide tremendously what you think you see and how you interpret the situation-- for complicated things or even easy things like lines or squares.

And here's another kind of an example where you would interpret that as a B for "baker" or 13 if it's in numbers. Again, the context is driving a lot of the interpretation. OK.

Now this is one of those examples that, again, when-- some number of years ago, it was a huge hit. And now, mostly people say, can't you come up with something better that we haven't all seen on the internet? So if you know this, don't ruin it for the other individuals. But what I need is a few volunteers-- you'll be facing me this way-- who are willing to count something. And it's MIT, we're pretty good at counting.

So what's the message of that? The message is-- we've talked about what we perceive, what we see by expectations in context. But it's also we have very limited what psychologists call attentional resources. We can pay attention to a limited number of things at a time. And even when those things can be right in front of us, if our attention is focused or occupied by something else, like counting the passes in a difficult scene-- it wouldn't work if there was one or two passes only, because you would notice it. But when your mind is focused on identifying all the passes among the players-- and the white shirts are moving, they're weaving with the other players and so on-- then your attention is absorbed by that, and some of it is not left over to notice what's right in front of you.

And we'll talk more about that. But it's a huge thing with humans that we can pay attention pretty well, on average, to a thing at a time under many circumstances. And the other things escape us completely, even if they're obviously present if we were looking at them or paying attention to them.
""")

writer.add_document(course_name=u"Intro to Psychology Fall 2011", segment_id=u"3",
transcript=u"""So here's another example of how our minds make our world-- what we see and what we don't see, what we pay attention to and what we don't pay attention to. And that's something to do with how we hear.

OK, so I'm going to replay this. So listen to what the guy is saying. Take a look, and just tell he-- he's saying some letters, OK, just not a word. What is it? OK, most people think he's saying "da." "Da da, da da, da da." Now let's try that again. I'm going to turn off the sound and I'm going to run the same film. What does his mouth look like it's saying? "Ga ga." OK?

But now we'll do one more thing, which is turn the sound back on, have you close your eyes, and listen to what he's saying. What's he saying?

AUDIENCE: "Ba."

PROFESSOR: Yeah. So it doesn't work for everybody every time. But the basic idea is most people think they hear the word "da" coming from the speaker. And in fact, in their mind they do because that's how they interpret what they're hearing. But in reality, the film clip is a film clip of the person saying "ba ba ba." And then an audio recording of the person saying "ga ga ga." Your mind intertwines across modalities what you hear and what you see, integrates them in some way below your level of consciousness. You're not thinking about it. And you come up with a different interpretation of what you hear. Right?

So what you see would be this one thing. What you hear is another thing. When your eyes are open and your ears are open, they meld together and produce something-- a third thing that's entirely different. Again, your mind interpreted what you hear, not your ear interpreting what you hear, in a simple sense.

OK. How about things that we know? So let's think about this. If somebody were to ask you which is farther east, closer to the Atlantic-- San Diego, California, or Reno, Nevada? Who likes San Diego as being farther east? A few hands. Who likes Reno as being farther east? OK. So, here's the mental map most people have-- the mental map-- which is we know California's right next to the ocean with Arnold Schwarzenegger protecting us on that side of the country, right?

And then Nevada's a little bit more towards Boston, right? OK. That's a mental map that most people have. And that's how the hands went up. This is the actual map. And the only actual map you've ever seen, ever-- on a globe, on a map, anything. Because California takes a big turn on the south, San Diego's further east than Reno.

Why do we imagine, and most people do, that Reno is further east, when you've never seen a map or globe that's shown you that? Never ever, ever. Yeah.

PROFESSOR: Because it's farther from the ocean, because in our mind we go, California's way out there. There's nothing-- Hawaii is the only one out there further west, right? So our mind makes this answer despite that. And that's what we think we might know. Now, we might not be totally certain. We might not bet the farm on that.

Which is farther north-- Philadelphia, Pennsylvania, or Rome, Italy? So start to think- - how would you think about that? It's not something you know. Nobody memorizes it, right? But how would you begin to think which is probably more northern? What's your first gut? How many people like Philadelphia being more north? How many people like Rome being more north? There's kind of a mixture of hands.

The answer is that Rome is north of Philadelphia. Mostly people will answer that Philadelphia is north. Why they do that is they think the US and Europe, they're both sort of above the equator, below Antarctica, kind of a aligned, even historically, culturally. So they think, well, Rome is pretty south in Europe. And it is. It's in Italy. Philadelphia's reasonably north in the US. It gets winters and all that kinds of stuff. So a northern city in the US has got to be north of a southern city in Europe. But in fact, Europe is-- the whole continent is shifted up compared to the US.

So you won't-- wait until get your mind around this. Which is further north, Atlanta or Chicago?

[LAUGHTER]

PROFESSOR: All right, all right. Sorry. It's sort of a joke. Because sometimes when you do this, people go like, wait a minute, all my assumptions are off. Like, where am I? What's reality?

[LAUGHTER]

PROFESSOR: OK. Here's one more-- two more. Which is further north, Portland or Toronto? Now you are already learning the lesson go opposite. Whatever I thought, go opposite, right? But why do you think most people will answer that Toronto is further north? Canada is up there, US is below it, but in fact-- that's the mental map in the colors. But in fact, Portland in Oregon is actually north of Toronto.

We'll do one last one. Which is further west? Which is further west, Miami, Florida-- which that's all the way towards the Atlantic Ocean-- or Santiago, Chili-- which is towards the Pacific Ocean. Further west. So most people have a mental map that North America and South America are kind of lined up like that. And so you say well, Miami is further east and Santiago's farther west. But in fact, South America is fairly shifted compared to North America. And Santiago is actually more eastern or Miami is more western, one relative to the other.

Because in our head, we kind of think, North and South America-- they're kind of lined up even though we never saw a global map like that. So again, some of our knowledge guides how we think about the world and what we believe we know.

So what's the point of this? It's what used to be called telephone, right? Their story keeps changing. And it's hard to remember details in a story. People remember a nugget, or what we call a gist in psychology, a little point. And second, what you take as a point is how you then tell the next person, the way you interpret the story, something like that. Thanks very much, that was good.

[APPLAUSE]

PROFESSOR: Again, two things-- our memory for precise details is surprisingly modest. And how we interpret things matter changes things a lot. So now, you had four brave students demonstrating some of the limits and properties of memory.

So now, here's an exercise you can do in your own seat. OK, you're just knowing yourself how you did, but here we go. I'm going to read you some words. And then just give you-- don't have to write anything down. If you write it down, it's no good. And then I'm going to ask you on a recognition test, whether you heard a word or not. Ready?

So here's the list. So just listen and then I'll test your memory for it right after. Here's the list. Sour, candy, sugar, bitter, good, taste, tooth, nice, honey, soda, chocolate, heart, cake, tart, pie. OK? All right, how many people heard the word "sour?" All right. Yeah, excellent, thank you. "Chair." "Candy." Hey. "Honey." "Building." "Sweet." Every hand up there, you have a false memory.

[LAUGHTER]

PROFESSOR: Now, it's a set up OK? Because here's the way they make these lists, it's a set up, but there's a huge lesson. And in fact, you may hear debates about what are real memories, what are false memories, in court cases, in clinical cases. This is a laboratory experiment that's been the testing ground for lots of ideas about how we make real memories and how we end up with false memories.

So here's the way they made the list. They took the word "sour." And they took a lot of students basically like you and said, what's the first word you think of that goes with sour? And people came up with this kind of a list. Candy, sugar, bitter, good, taste, tooth, nice, honey, soda, chocolate, heart, cake, tart, pie. But they left out one word that people came up with a lot. The word "sweet." OK?

So your mind interpreted the list. You said, hey, this is all about things that are related to sweet things in one way or another-- sweet sugar, sweet candy, sweet and sour, honey is sweet, chocolate is sweet. So your mind imagined it heard the word "sweet." And the majority of you put your hand up that you actually heard the word "sweet." Your mind imagined it was there because that was generally what was going on. That was the gist of the experience, OK?

So this idea is it's very easy, because of the way memory works, we remember the gist of things because that's what's the important part. It's hard to remember the details. But that gist is an interpreted gist. The gist was it's sweet things. So the word "sweet" feels like it was part of the memory. And we'll come back to that later on in the course.
""")

writer.add_document(course_name=u"Intro to Psychology Fall 2011", segment_id=u"4",
transcript=u"""So one of the themes we'll talk about a lot in the course is both an amazing power of the human mind and an amazing peril of the human mind. And it's what psychologists call automaticity. It's that our mind, in order to be efficient and quick, does things automatically without thought, without consciousness. It lets us walk without thinking a lot about where our feet are. It lets us speak quickly without thinking about the syntax and the vocabulary, right? It lets us do a lot of things. So that's the power of it.

The peril is when something becomes automatic, we lose control of it within ourselves. So I need somebody at their seat who's willing to read aloud something as fast as they can when they see it on the computer monitor. If I can get a volunteer at your seat. OK, all the way back there, OK. And then I'll come to you for the second one. Ready? Here it comes. As fast as you can, go.

AUDIENCE: One way not do enter.

PROFESSOR: OK, then, you got it. I couldn't trick you. OK. But you might imagine a person might mistake that, right? Was there another one? Was it you? OK, ready? Here we go. Go.

AUDIENCE: Paris in spring.

PROFESSOR: Ah. I got you on that.

[LAUGHTER]

PROFESSOR: Because your mind is automatically reading. We have lots of evidence in psychology that you're barely looking at words like "the." You're assuming over those things. They're almost invisible to you there even though they're physically present, because your mind is looking for the big content, right? Who cares about the word "the?" Your mind is going for the essential information, and it becomes literally blind to what's in front of you, because it knows what it's looking for.

Here's a fun one. You've seen things like this before, but it's always fun to try. It's the same principle. How many letter F's do you find in this display? Can I get some numbers?

AUDIENCE: 6.

AUDIENCE: 4.

AUDIENCE: 5.

PROFESSOR: 4, 5, 6. Those are all good. We're not an exact science.

[CLAMORING]

PROFESSOR: Some of you may have missed one or two F's. Again, it's because your mind is automatically-- typical readers read at spectacular speeds. And the way you read at a spectacular speed is you don't look for little details. You get the big words and the big ideas and you zoom through for the big meaning. And you're leaving behind what you consider to be details. Yeah.

AUDIENCE: So if you ask this question to a society that pronounces "of" just like "off," would that change anything?

PROFESSOR: The question was if we asked a society that didn't pronounce F's or something like that.

AUDIENCE: That didn't pronounce F's as F's. In America, we pronounce it "of."

PROFESSOR: "Of," you mean like a "v" sound or something like that. Does that matter for this? Yes. It also matters a lot that words like "of" are little preposition words that we don't think much about. So this is a set up. Like "finish," most people get. Or the beginning of a word you're more likely to get. I think the pronunciation probably matters. I don't know that for sure. That's a very good thought. And certainly, hiding it in words that seem low in content for interpreting a sentence is about the best way we did it. That's why the second "the" disappeared too. It's sort of a low content word for processing a sentence.

OK. This is an example that you know, but it's a nice example and we can come back to it a couple times. So let me think about this for one second. Maybe we'll do it this way-- that we'll ask somebody at their seat who has typical color vision. If you're color blind, this one is not a good one for you. Some percentage are.

Is somebody willing at their seat to read aloud stuff they see on a monitor? OK, thank you. Here we go. So you're gonna see words that are printed in different colors. Your job is to name aloud the color of the ink that it's printed in. Does that make sense? So like on this F, you would say it's red on that F. Is that OK?

Here we go. So start here and just go.

AUDIENCE: Red, orange.

PROFESSOR: As fast as you can, just keep going.

AUDIENCE: Green, brown, pink, green, blue, yellow, red.

PROFESSOR: Great, excellent. Same thing. Read the color of the ink exactly like you were doing. Go.

AUDIENCE: Green, blue, red, blue, red, yellow, red.

PROFESSOR: Ah, you're pretty good. OK. It's supposed to slow you down when you get the ink in the wrong colors. And it usually does. But you were very good.

All right. Again, if you know this from courses and the internet, don't ruin it for others, but think about it for yourself. So now we're gonna turn to thought. There's 30 people in a room. Just imagine you sat-- there' just groups of 30 here. You get the month and date of each person's birthday. So it's not the year they were born, but it could be December 1 or February 5 or something like that. What is the approximate probability that two people will have the exact same birthday?

I can tell you the vast majority of people under slightly less suspicious circumstances of this will answer about 10%. That's the vast majority. The correct answer is-- OK? Why do you think-- this is work from Kahneman and Tversky. We'll come back to this. Why do you think people tend to answer 10%, some 30%? Very few people give you the mathematically correct answer of 70%. Why do they do that?

Because they tend to think, how often have I met somebody who has my exact birthday? And you go, not that often. It's not like every 30 people I meet, somebody says, you were born on March 3. I was born in March 3. And then you go have lunch and you go, hey, I was born on March 3. And you go have dinner with another group and they go, I was born on March 3.

It's not something that happens a lot, right? So you go, well, in real life it doesn't seem to happen very often. That's what we call a heuristic-- a simple way to think about it. Because your experience is kind of like that. But why is that incorrect mathematically for this question? Because the math depends on not that's exactly your birthday, but any pair of birthdays among the 30 people.

And then it goes way up. In fact, it goes to 70%. And if it's 24 people, it's 50%. If you're a group of 36 people, there's a 90% chance, just mathematically, that two people will share the same birthday. Because when we face things that are hard to think about, because there's no easy answer, humans tend to take shortcuts and say, what's the gist of my experience, and that's what I think the answer is. Even when a calculable answer is available. It's human nature to make a shortcut based on your sense of your experience.
""")

writer.add_document(course_name=u"Intro to Psychology Fall 2011", segment_id=u"5",
transcript=u"""So there's a very interesting line of work-- Dan Gilbert of Harvard is a leading figure- - about this idea of thinking about your future. Now, thinking about our future is a big thing, right? We're thinking about what's it like in this course, what's it like in college, what's our friendship like, relations with parents, what's our future career paths, what kind of life will we lead, right? Our future is something that's hugely on our mind, I think, very powerfully when you're a college student or a graduate student. What's my future?

And a big question that people have is what will make me happy in a deep sense? What will make me happy in a deep sense? Because that's the life I want to lead-- the values I want to have, the kind of career choices and personal choices I want to make, where I will devote my time on this earth.

So most people, first of all, tend to think about good things, positive things. Actually, I can tell you what comes later in the course. It's good to think that lots of positive things are happening. It's kind of a nice place to be in terms of being a happy person. But it turns out that people have done studies like this. So now this is particularly sensitive for a faculty member, but it could work for any sports team you've tried out or anything you've tried out for in your life.

So what happens when we get reviewed for tenure? And you hear a bit about that. This was an easy study for a psychologist to do. What they did is they called up people in the fall who were being reviewed for tenure. And you get tenure or you don't. And it's a bit of a sad process if you don't, right, because you don't get tenure, and then you don't feel happy about that. And you have to call your parents and say, I didn't get tenure, and your parents go, come on, if you just slept better, you would've gotten tenure.

[LAUGHTER]

PROFESSOR: Remember the piano lessons you didn't take. So it's a bit of a nuisance, right? On top of that-- because weirdly, in academics, we tend to be super specialized-- you have to move out of town. You don't have to, but typically, a person who doesn't get tenure will get a job somewhere else. There's plenty of stories of people who don't get tenure at awesome places who were geniuses in history. The tenure decisions are often wrong. But still, you'd rather get it than not. You'd rather get into the medical school than not. You'd rather make a sports team you want to be on than not.

So here's what they found out. If they asked them what happens if you don't get tenure, everybody says, oh, it's gonna be awful. It's gonna be miserable. I'm gonna be such an unhappy person. Two years later, the average happiness of people who didn't get tenure was equal to the average happiness of people who did get tenure.

So you can say, well, tenure-- only professors care about tenure. Well, how about winning the lottery? What if I won hundreds of thousands of dollars? There' been a lot of psychology on this, actually. In about a year to two, the average happiness of a lottery winner who won a substantial amount of money is rated the same by him or her as it was the population as a whole. Yeah?

AUDIENCE: How did they go about measuring average happiness?

PROFESSOR: Yeah, so we'll come back to this, but I'll tell you. You can like this or not like this. In some parts of psychology, we measure things like reaction time to the millisecond. That's good data, right? Our brain activation, that's good data. When you ask a person how happy they are, the only thing we can do is have you basically fill a scale from one to seven. How happy are you? And you could go, well I'm a little worried about that, because sometimes people say, I hope that it makes you happy or something. So you could say, how much can we trust subjective reports of happiness? And that's a very good question.

On the other hand, it's hard to know what would be better than that. If we measure your pulse, is that a better measure of happiness? Your pulse could be racing because you're sad or happy, scared or enthusiastic. So we don't have a better one that we can think of. But psychologists do worry that sometimes people will just say what they're supposed to say. Or they'll pretend they're happy or things like that. We have to worry about those things.

So you could worry deep down, but a year or two later, people who win huge amounts of money don't report themselves as any happier than people around them. And kind of amazingly-- but I think it's deep about life-- accidents leading to quadriplegia or paraplegia, accidents that, before you had such an accident you would imagine that it would be something extremely difficult. And it can be in many ways. But by self report, ratings of happiness return to typical average populations of the same age in about three months.

So what's a huge lesson here in happiness research-- a huge surprise. It's two things. We're kind of bad at predicting what will make us happy or sad, which is kind of weird, right? We're kind of bad at predicting it. Here's all these things where we think they would make us happy or make us not so happy. It turns out we're wrong when this is studied at all scientifically. So we'll come back to that later on, because it's a very deep thing about being a human-- what makes you happy and your wrong guesses sometimes about what does.

So let me end with a last experiment. So we've really haven't done experiments until right now. And this is now a sensitive and difficult issue, which is problems we have in dealing with racism. And here's a study that did the following. It said, well, in North America, certainly, Canada, the US-- a study was done in Canada-- racism is widely condemned, as I think most of us believe it should be. But examples of blatant racism still occurred. One recent poll said that about a third of white individuals reported hearing anti-black slurs in the workplace in the last couple years-- to pick one thing.

So how does this happen in a society that speaks so much about not being racist, about treating everybody equally and fairly and kindly? How does it happen that we still struggle. And it's such a very deep, difficult question about human nature and the world we live in. But here's something again that's a hint about why it's hard to get society to change some of its behaviors.

So here's the experiment-- so it's an actual experiment. So they took two groups of college undergraduates and randomly assigned one to be in the forecaster group. That's a group that tells you how they think they would feel and how they think they would act under certain circumstances. And then an experiencer group-- that's a group who actually undergoes an experience, and I'll tell you what that is now.

So in the experiencer group, pretend you were their research participant. You walk into a room, and you see in that room a black male and a white male. Now those two are what psychologists, for some reason, have called confederates. Those are role players. They know what they're doing. They have a plan of what they're going to do. They're going to put on a little show for you. But you don't know that.

And the black male stands up and leaves the room to get his cell phone, and he gently bumps the white male's knee. This is all set up. You're just sitting there and you see that little bump. And now, there's three different groups. One group, that's it. Nothing else happens-- A small bump, and the person leaves. A second group-- as you sit there, the black individual leaves the room and the white individual says, quote, "Typical, I hate it when black people do that." It's meant to be obviously provocative and racist. And then what they consider an extreme slur-- the white person in the room playing this role uses the derogatory word that's meant to be an extreme slur.

So there's one more thing you need to know. Now, you're sitting there, and you're either in the control group where there's been the slight bump, or there's been a moderate slur, or an extreme slur in their words. The black male returns. Don't forget, he's in on it, and so is that white male. But you're not in on it. You just think there was a bump, and something else may have happened, depending on which condition you're in.

And the experimenter then gives you a survey about how you feel right now. Sort of like the happiness, but it's not that. It's like, how do you feel right now? And then asks you to pick between those two people a partner for an anagram experiment that you're about to do.

So they're going to ask you-- this is sort of this question you have. What's the difference or similarity between what you say you're feeling is and what you really do? Both things are important, but do they line up, do they not line up?

So here's the results. Here's a graph. And here's how this works. Negative emotional distress the higher the bar, the more you say, I feel really bad about what's just happened. I just heard this comment or no comment. So let's take a look, the higher the bar. If you heard no comment, here's how you begin.

So let's start with the forecasters. All of you are forecasters, because you're pretending you're in the situation but you're not in it. So here is there was no comments. That's sort of average or something. And then you said if you heard a racial slur you would feel terrible. You would feel terrible.

But look at the other students who are randomly picked. So we don't think it's a difference among students. Look at these grey bars. They're pretty flat. The person on the spot is somehow not processing this. And they're filling out, I feel average.

You see the split-- the split between the values that the person thinks they would have, and the values that are responded to on the spot in the moment. And what we'll talk about later on in social psychology is there's a tough gap, often, between the values we espouse and how we act when there's especially unexpected, difficult things.

And very often-- if you've had any experience like this-- afterwards, you go, oh, what I should have done is this. Or I wish I would've said that. But that moment is not happening at that moment, probably because you're kind of weirded out by the whole thing. What's going on? Why would the person say this? Something doesn't seem right. I can't sort it out. And so people tend to shrink in terms of making a strong conclusion of what's going on if something seems unusually provocative.

And you could say, well, OK, that's their attitudes. But how about their action? Who do they pick to be their partner? And again, the people forecasting said, if I was in this situation, I would never pick that racist white person to be my partner, because that person stinks-- if I was in that situation. But if the people are in the situation-- look at the grey bars-- pretty flat. It's a if on the spot, in the moment, they can't quite process the values they feel and the action they're going to take.

And we'll talk about that. And it's very hard, often, in part, to be brave and stand up to things. It turns out there's a lot of evidence for this. It's a human nature thing. It's very hard to be brave and stand up to things when things are kind of weird, because almost everybody at first thinks, I don't want to make a fool of myself. I don't want make trouble. Maybe I'm not getting the whole picture on this. And we shrink back from acting in a way that aligns with the values that are clearly shown here.

So this, again, is something about human nature that's very weird. And it's powerful to come into social psychology. And that's why it's very hard to stand up to things like oppression and bias. It's very hard to do, because we tend to not act on our values when we're in complicated situations on the spot. And there's a tremendous amount of evidence for that.

So again, how we interpret the situation-- very different in our mind when we imagine we're there, and when we actually sit there. And so what these researchers say is this is partly why it's been hard to eradicate some vestiges of stereotypes and racism, because people have a hard time clamping down on it in the moment. So that's a tough topic, but we know we want to deal both with things that are less controversial but also things that touch people's lives in the real world that we live in.

So we talked about a scientific study of the human nature-- mind and behavior-- how what we see and hear is determined so much but how our mind interprets the world around us; how we remember things like word lists or stories, that's hugely influenced by what we expect to see, like in the picture; how we think we know things like where Reno is compared to San Francisco; how we think about things like the probability that somebody else will have the same birthday, that somebody else will in a group; and the relationship between how we feel and how we act. The very feelings we have are often disconnected for actions in the moment. And sometimes that has a sort of a difficult consequence.

And so we'll explore all these things through the semester, all the different facets that we could possibly get through in one semester of what it is to be human, and where science has showed us something about human nature, the mind, and the brain.
""")

writer.commit()
