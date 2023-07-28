from search import SemanticSearch


test_set = [{"pdf": """Deep neural networks (DNNs) are often used
for text classification tasks as they usually
achieve high levels of accuracy. However,
DNNs can be computationally intensive with
billions of parameters and large amounts of labeled data, which can make them expensive
to use, to optimize and to transfer to out-ofdistribution (OOD) cases in practice. In this
paper, we propose a non-parametric alternative
to DNNs that’s easy, light-weight and universal in text classification: a combination of a
simple compressor like gzip with a k-nearestneighbor classifier. Without any training, pretraining or fine-tuning, our method achieves results that are competitive with non-pretrained
deep learning methods on six in-distributed
datasets. It even outperforms BERT on all five
OOD datasets, including four low-resource
languages. Our method also performs particularly well in few-shot settings where labeled
data are too scarce for DNNs to achieve a satisfying accuracy""", "query": "What is a possible way to classify text?"},
            {"pdf": """Removing background noise from speech audio has been the subject of considerable effort, especially in recent years due to the rise of virtual communication and
amateur recordings. Yet background noise is not the only unpleasant disturbance
that can prevent intelligibility: reverb, clipping, codec artifacts, problematic equalization, limited bandwidth, or inconsistent loudness are equally disturbing and
ubiquitous. In this work, we propose to consider the task of speech enhancement as
a holistic endeavor, and present a universal speech enhancement system that tackles
55 different distortions at the same time. Our approach consists of a generative
model that employs score-based diffusion, together with a multi-resolution conditioning network that performs enhancement with mixture density networks. We
show that this approach significantly outperforms the state of the art in a subjective
test performed by expert listeners. We also show that it achieves competitive
objective scores with just 4–8 diffusion steps, despite not considering any particular
strategy for fast sampling. We hope that both our methodology and technical
contributions encourage researchers and practitioners to adopt a universal approach
to speech enhancement, possibly framing it as a generative task.""", "query": "What is a possible way to enhance speech in audio?"},
            {"pdf": """Abstract. Image-based visual-language (I-VL) pre-training has shown
great success for learning joint visual-textual representations from largescale web data, revealing remarkable ability for “zero-shot” generalisation. This paper presents a simple but strong baseline to efficiently adapt
the pre-trained I-VL model, and exploit its powerful ability for resourcehungry video understanding tasks, with minimal training. Specifically, we
propose to optimise a few random vectors, termed as “continuous prompt
vectors”, that convert video-related tasks into the same format as the pretraining objectives. In addition, to bridge the gap between static images
and videos, temporal information is encoded with lightweight Transformers stacking on top of frame-wise visual features. Experimentally, we conduct extensive ablation studies to analyse the critical components. On
10 public benchmarks of action recognition, action localisation, and textvideo retrieval, across closed-set, few-shot, and zero-shot scenarios, we
achieve competitive or state-of-the-art performance to existing methods,
despite optimising significantly fewer parameters.""", "query": "What is a possible way to efficiently understand videos?"},
            {"pdf": """Signal processing (SP) excels at analyzing, processing, and inferring information defined over
regular (first continuous, later discrete) domains such as time or space. Indeed, the last 75 years
have shown how SP has made an impact in areas such as communications, acoustics, sensing,
image processing, and control, to name a few. With the digitalization of the modern world and the
increasing pervasiveness of data-collection mechanisms, information of interest in current applications
oftentimes arises in non-Euclidean, irregular domains. Graph signal processing (GSP) generalizes SP
tasks to signals living on non-Euclidean domains whose structure can be captured by a weighted
graph. Graphs are versatile, able to capture irregular interactions, easy to interpret, and endowed
with a corpus of mathematical results, rendering them natural candidates to serve as the basis for
a theory of processing signals in more irregular domains.
The term “graph signal processing” was coined a decade ago in the seminal works [1], [2], [3],
[4]. Since these papers were published, GSP-related problems have drawn significant attention,
not only within the SP community [5], [6] but also in machine learning venues, where research in
graph-based learning has increased significantly [7]. Graph signals are well-suited to model measurements/information/data associated with (indexed by) a set where: (i) the elements of the set belong
to the same class (regions of the cerebral cortex, members of a social network, weather stations""", "query": "How to process graph signals?"},
            
            {"pdf": """Traditional computational fluid dynamics calculates the physical information of the flow field by solving
partial differential equations, which takes a long time to calculate and consumes a lot of computational
resources. We build a fluid simulation simulator based on the graph neural network architecture. The
simulator has fast computing speed and low consumption of computing resources. We regard the
computational domain as a structural graph, and the computational nodes in the structural graph
determine neighbor nodes through adaptive sampling. Building deep learning architectures with attention
graph neural networks. The fluid simulation simulator is trained according to the simulation results of
the flow field around the cylinder with different Reynolds numbers. The trained fluid simulation
simulator not only has a very high accuracy for the prediction of the flow field in the training set, but
also can extrapolate the flow field outside the training set. Compared to traditional CFD solvers, the fluid
simulation simulator achieves a speedup of 2-3 orders of magnitude. The fluid simulation simulator
provides new ideas for the rapid optimization and design of fluid mechanics models and the real-time
control of intelligent fluid mechanisms.""", "query": "What is a possible way to simulate fluid?"},
            {"pdf": """We study the class of compact spaces that appear as structure spaces of separable Banach lattices. In other words, we analyze what C(K) spaces appear as principal
ideals of separable Banach lattices. Among other things, it is shown that every such
compactum K admits a strictly positive regular Borel measure of countable type that is
analytic, and in the nonmetrizable case these compacta are saturated with copies of βN.
Some natural questions about this class are left open.""", "query": "What is a lattice?"},
            {"pdf": """Abstract—Self-supervised skeleton-based action recognition
with contrastive learning has attracted much attention. Recent
literature shows that data augmentation and large sets of contrastive pairs are crucial in learning such representations. In
this paper, we found that directly extending contrastive pairs
based on normal augmentations brings limited returns in terms
of performance, because the contribution of contrastive pairs
from the normal data augmentation to the loss get smaller as
training progresses. Therefore, we delve into hard contrastive
pairs for contrastive learning. Motivated by the success of mixing
augmentation strategy which improves the performance of many
tasks by synthesizing novel samples, we propose SkeleMixCLR:
a contrastive learning framework with a spatio-temporal skeleton mixing augmentation (SkeleMix) to complement current
contrastive learning approaches by providing hard contrastive
samples. First, SkeleMix utilizes the topological information
of skeleton data to mix two skeleton sequences by randomly
combing the cropped skeleton fragments (the trimmed view) with
the remaining skeleton sequences (the truncated view). Second,
a spatio-temporal mask pooling is applied to separate these two
views at the feature level. Third, we extend contrastive pairs
with these two views. SkeleMixCLR leverages the trimmed and
truncated views to provide abundant hard contrastive pairs since
they involve some context information from each other due
to the graph convolution operations, which allows the model
to learn better motion representations for action recognition.
Extensive experiments on NTU-RGB+D, NTU120-RGB+D, and
PKU-MMD datasets show that SkeleMixCLR achieves state-ofthe-art performance. Codes are available at https://github.com/
czhaneva/SkeleMixCLR.""", "query": "What is a good way to do action recognition?"},
            {"pdf": """This paper studies the quantum computational complexity of the discrete logarithm and
related group-theoretic problems in the context of “generic algorithms”—that is, algorithms
that do not exploit any properties of the group encoding.
We establish a generic model of quantum computation for group-theoretic problems, which
we call the quantum generic group model, as a quantum analog of its classical counterpart.
Shor’s algorithm for the discrete logarithm problem and related algorithms can be described in
this model. We show the quantum complexity lower bounds and (almost) matching algorithms
of the discrete logarithm and related problems in this model. More precisely, we prove the
following results for a cyclic group G of prime order.
• Any generic quantum discrete logarithm algorithm must make Ω(log |G|) depth of group
operation queries. This shows that Shor’s algorithm that makes O(log |G|) group operations is asymptotically optimal among the generic quantum algorithms, even considering
parallel algorithms.
• We observe that some (known) variations of Shor’s algorithm can take advantage of classical computations to reduce the number and depth of quantum group operations. We
introduce a model for generic hybrid quantum-classical algorithms that captures these
variants, and show that these algorithms are almost optimal in this model. Any generic
hybrid quantum-classical algorithm for the discrete logarithm problem with a total number of (classical or quantum) group operations Q must make Ω(log |G|/ log Q) quantum
group operations of depth Ω(log log |G| − log log Q). In particular, if Q = poly log |G|,
classical group operations can only save the number of quantum queries by a factor of
O(log log |G|) and the quantum depth remains as Ω(log log |G|).
• When the quantum memory can only store t group elements and use quantum random
access memory (qRAM) of r group elements, any generic hybrid quantum-classical algorithm must make either Ω(p
|G|) group operation queries in total or Ω(log |G|/ log(tr))
quantum group operation queries. In particular, classical queries cannot reduce the number of quantum queries beyond Ω(log |G|/ log(tr)).
As a side contribution, we show a multiple discrete logarithm problem admits a better algorithm than solving each instance one by one, refuting a strong form of the quantum annoying
property suggested in the context of password-authenticated key exchange protocol.""", "query": "What is a good way to solve the discrete log problem?"},
            {"pdf": """Black holes have always been a fertile area of research with regards to the quantum theory of
gravity. The holographic principle motivated by string theory have made possible advances in our
understanding of how black holes scramble in fallen information. A quantum information theoretic
approach for large N field theories had allowed Hayden, Preskill and others [1, 2] to estimate
the fastest time scales possible to scramble any local1 perturbation among its microstates to be
t∗ ∼ log S, S being the entropy of the system. These works also conjectured that black holes are
amongst the fastest scramblers of in fallen information in nature. A precise measure of such a chaotic
behaviour was fist performed by checking the rate of scrambling due to a O(GN ) perturbation of
the finely tuned entanglement between the 2 CFTs corresponding to the Thermo-field Double(TFD)
state dual to a Schwarzchild black hole in AdS3 [3]. It was also shown that the 4pt out of time
ordered correlators (OTOCs) which are a good measure of chaotic phenomena, see an exponential
growth in their first sub-leading term in GN at large time separations""", "query": "What are black holes?"},
            {"pdf": """Abstract. Higher-order networks are efficient representations of sequential data. Unlike the classic first-order network approach, they capture
indirect dependencies between items composing the input sequences by
the use of memory-nodes. We focus in this study on the variable-order
network model introduced in [12
,10]. Authors suggested that randomwalk-based mining tools can be directly applied to these networks. We
discuss the case of the PageRank measure. We show the existence of
a bias due to the distribution of the number of representations of the
items. We propose an adaptation of the PageRank model in order to
correct it. Application on real-world data shows important differences in
the achieved rankings.""", "query": "What are higher order networks?"}
            ]
documents = [x['pdf'] for x in test_set]
search_engine = SemanticSearch(documents)

count = 0
for i, x in enumerate(test_set):
    result = search_engine.search(x['query'], k=3)
    if documents[i] in result:
        count +=1

print(count / len(test_set))
