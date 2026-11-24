# Introduction to Learning with Limited Labeled Data

In many real-world applications of supervised learning, the availability of labeled data is often limited due to the high cost or difficulty of obtaining expert annotations. This scarcity of labeled data poses a significant challenge to building accurate and robust models. To address this, researchers and practitioners have developed several strategies that aim to maximize the utility of the limited labeled data while leveraging the abundance of unlabeled data. These strategies form the foundation for effective learning in low-data regimes and include pre-training plus fine-tuning, semi-supervised learning, active learning, and pre-training with dataset auto-generation. This article explores these approaches in depth, with a particular focus on semi-supervised learning, which is a powerful paradigm for improving model performance when labeled data is scarce.

## Pre-training Plus Fine-tuning

One of the most widely used strategies for dealing with limited labeled data is pre-training a model on a large amount of unlabeled data and then fine-tuning it on the small labeled dataset. This approach is particularly effective in domains such as natural language processing and computer vision, where pre-trained models like BERT or ResNet have demonstrated strong performance across a range of tasks. By leveraging the rich representations learned during pre-training, fine-tuning allows the model to adapt to the specific characteristics of the target task with minimal labeled data [citation:txt16].

## Semi-Supervised Learning

Semi-supervised learning explicitly incorporates both labeled and unlabeled data into the training process. This approach is especially valuable when unlabeled data is abundant but labeled data is scarce. Semi-supervised methods often rely on assumptions such as the cluster assumption or the manifold assumption, which posit that data points close to each other in the input space are likely to share the same label. These assumptions guide the learning process and enable the model to make more informed predictions about the unlabeled data [citation:txt5]. Techniques like pseudo-labeling, consistency regularization, and graph-based methods have been developed to effectively utilize unlabeled data within a semi-supervised framework [citation:txt1].

## Active Learning

Active learning is another strategy that addresses the issue of limited labeled data by intelligently selecting the most informative samples for labeling. Instead of randomly sampling data points, active learning algorithms query the model to identify instances where the model is most uncertain. By focusing labeling efforts on these uncertain instances, active learning can significantly reduce the amount of labeled data required to achieve a given level of performance [citation:txt19]. This approach is particularly useful in domains where human annotation is expensive or time-consuming.

## Pre-training Plus Dataset Auto-generation

A more recent and innovative strategy is pre-training combined with dataset auto-generation. In this approach, a pre-trained model is used to generate additional labeled data, which is then used to fine-tune the model further. This has been particularly effective in the language domain, where pre-trained models can generate synthetic examples that are used to expand the training dataset. This technique is driven by the success of few-shot learning, where models can perform well on new tasks with only a small number of examples [citation:txt6].

Together, these four strategies provide a comprehensive toolkit for learning with limited labeled data. While this section has introduced the main approaches, the focus of the article is on semi-supervised learning, which has seen significant recent advances and remains a central topic in the field of machine learning.

# Foundations of Semi-Supervised Learning

Semi-supervised learning (SSL) is a paradigm that leverages both labeled and unlabeled data to train a model, aiming to improve performance when labeled data is scarce. In this setting, the labeled dataset typically contains a small number of examples with known class labels, while the unlabeled dataset consists of a larger number of examples without labels. The goal of SSL is to extract additional information from the unlabeled data to enhance the model's generalization capabilities beyond what is possible with labeled data alone [citation:txt9].

Mathematically, let us define the labeled dataset as $D_L = \{(x_1, y_1), \dots, (x_l, y_l)\}$ and the unlabeled dataset as $D_U = \{x_{l+1}, \dots, x_{l+u}\}$, where $x_i \in \mathcal{X}$ is an input feature vector and $y_i \in \mathcal{Y}$ is the corresponding label. The total dataset is then composed of both labeled and unlabeled examples, $D = D_L \cup D_U$. The model $f_\theta$ is trained to approximate the underlying mapping from $\mathcal{X}$ to $\mathcal{Y}$ [citation:txt16].

The training objective in SSL is typically expressed as a combination of a supervised loss and an unsupervised loss. The supervised loss is computed using the labeled data and aims to minimize the discrepancy between the model's predictions and the true labels. The unsupervised loss, on the other hand, is derived from the unlabeled data and is designed to enforce certain structural assumptions about the data distribution. The total loss function can be written as:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \lambda \cdot \mathcal{L}_{\text{unsupervised}},
$$

where $\lambda$ is a weighting factor that balances the contributions of the two components [citation:txt15].

The design of SSL methods is grounded in several key assumptions that guide the use of unlabeled data. These assumptions reflect the intuition that the structure of the input space can be exploited to make more informed predictions. The four most widely recognized assumptions are:

1. **Smoothness Assumption**: If two input samples $x$ and $x'$ are close in the input space, their corresponding labels $y$ and $y'$ should be the same. This assumption implies that the label function varies smoothly with respect to the input features. In other words, small perturbations in the input should not lead to drastic changes in the predicted label [citation:txt2].

2. **Cluster Assumption**: Data points tend to form clusters, and the decision boundary between classes should avoid slicing through these clusters. This means that samples within the same cluster are likely to belong to the same class. The cluster assumption is closely related to the idea that the class label should not change in regions of high data density [citation:txt10].

3. **Low-Density Separation Assumption**: The decision boundary between classes should lie in regions of low data density. This assumption is equivalent to the cluster assumption and is often used in consistency regularization methods, where the model is encouraged to produce consistent predictions for similar inputs [citation:txt17].

4. **Manifold Assumption**: The data lies on a low-dimensional manifold embedded in a high-dimensional space. The label function is smooth with respect to the structure of this manifold. This assumption is particularly useful in scenarios where the input space is high-dimensional but the data lies on a lower-dimensional subspace [citation:txt4].

These assumptions are not mutually exclusive and often overlap in practice. Many SSL algorithms implicitly or explicitly rely on one or more of these assumptions to guide the training process. For example, consistency regularization methods enforce the smoothness and low-density separation assumptions by encouraging the model to produce consistent predictions for different augmentations of the same input [citation:txt17].

In the next section, we will explore specific SSL techniques that build upon these theoretical foundations, including pseudo labeling, consistency regularization, and pre-training methods.

# Consistency Regularization in Semi-Supervised Learning

Consistency regularization is a key technique in semi-supervised learning that enforces model predictions to remain stable under input or model perturbations. The fundamental principle is that a model should produce consistent outputs for different augmentations of the same input. This approach leverages the low-density separation and smoothness assumptions, ensuring that the decision boundary traverses regions of low data density [citation:txt13]. By encouraging such invariance, consistency regularization helps the model avoid overfitting to noisy or limited labeled examples [citation:txt1].

One of the earliest consistency regularization frameworks is the Π-model, introduced by Laine and Aila (2017), which trains two different perturbations of the same unlabeled input and enforces their predictions to be similar. The loss function is typically the mean squared error or KL divergence between the outputs of the two models [citation:txt9]. Temporal ensembling extends this idea by maintaining an exponential moving average of past model predictions, which serves as a more stable target for consistency training. This approach smooths out noisy predictions over time, improving the model’s robustness [citation:txt9].

The mean teacher framework builds upon temporal ensembling by using a slow-moving "teacher" model to provide pseudo-labels for the "student" model. The teacher model is updated using an exponential moving average of the student’s weights, ensuring that it reflects a more stable and consistent version of the model [citation:txt15]. This method has been particularly effective in tasks where data augmentation plays a critical role, such as image classification and natural language processing.

Beyond early consistency methods, recent advances have introduced more sophisticated approaches, including adversarial and virtual adversarial training (VAT). VAT, proposed by Miyato et al., finds the worst-case perturbation that maximizes the model’s prediction change and regularizes the model to be consistent with respect to such adversarial examples [citation:txt15]. This method has shown strong performance in semi-supervised learning by explicitly targeting the most informative perturbations [citation:txt14].

Interpolation consistency training (ICT) is another notable approach that uses linear interpolations between pairs of unlabeled examples to generate additional training signals. By enforcing consistency between predictions on the original and interpolated samples, ICT encourages the model to learn smooth and continuous decision boundaries [citation:txt15]. This technique is particularly effective when combined with strong data augmentation strategies [citation:txt15].

Unsupervised data augmentation (UDA), introduced by Xie et al. (2019), is a powerful and widely adopted method that combines consistency regularization with advanced data augmentation techniques. UDA applies strong augmentations—such as RandAugment or AutoAugment—to unlabeled data and enforces the model to produce the same output for both the original and augmented examples [citation:txt2]. The loss is computed using the KL divergence between the model’s predictions on the original and augmented samples [citation:txt3]. UDA has demonstrated impressive performance on vision tasks, often matching or exceeding fully supervised models trained on much larger labeled datasets [citation:txt2].

Empirical studies have shown that the quality of data augmentation plays a crucial role in the success of consistency regularization. UDA, in particular, benefits from using diverse and realistic augmentations that preserve the underlying data distribution while introducing meaningful variations [citation:txt5]. For example, on the CIFAR-10 dataset, UDA outperforms previous methods by a clear margin when using state-of-the-art augmentation strategies [citation:txt2].

Recent work has also explored combining consistency regularization with pseudo labeling and pre-training. For instance, FixMatch [citation:txt14] integrates confidence-based pseudo labeling with consistency training, achieving strong results on multiple semi-supervised benchmarks. These hybrid approaches demonstrate the effectiveness of combining multiple SSL strategies to further improve performance [citation:txt15].

In summary, consistency regularization provides a principled way to leverage unlabeled data by enforcing prediction stability under input perturbations. Methods like the Π-model, temporal ensembling, mean teacher, VAT, ICT, and UDA have significantly advanced the field, with empirical results showing that strong data augmentation and consistency training can nearly match fully supervised performance [citation:txt2]. In the next section, we will explore how these techniques can be integrated with pre-training to further enhance semi-supervised learning.

# Pseudo Labeling and Iterative Self-Training in Semi-Supervised Learning

Pseudo labeling is a foundational technique in semi-supervised learning (SSL) where a model generates labels for unlabeled data based on its own predictions and then uses these pseudo-labels as if they were ground truth labels for further training. This method is particularly valuable when labeled data is scarce, as it enables the model to learn from a much larger pool of unlabeled examples. In its simplest form, pseudo labeling assigns the class with the highest softmax probability as the pseudo-label for each unlabeled sample [citation:txt1]. These pseudo-labels are then incorporated into the training set, allowing the model to iteratively refine its predictions [citation:txt6].

The process of pseudo labeling is closely related to self-training, where the model iteratively improves itself by generating and using pseudo-labels. This iterative approach is often visualized in algorithmic form, as illustrated in the following self-training procedure:

![](citation:img1)

In this procedure, the model is first trained on the small labeled dataset $D_l$, and then used to generate pseudo-labels for the unlabeled dataset $D_u$. Only the most confident predictions are retained as pseudo-labels and added to the training data for the next iteration. This cycle continues until a predefined number of iterations $T$ is reached, allowing the model to progressively improve its performance by leveraging increasingly accurate pseudo-labels [citation:txt12].

Label propagation is another related method that extends the concept of pseudo labeling by incorporating graph-based structures. In label propagation, a similarity graph is constructed between all data points, and labels are propagated through this graph to unlabeled samples based on their proximity to labeled instances. This approach ensures that unlabeled data points with similar features to labeled ones inherit their labels, effectively leveraging the smoothness assumption of SSL [citation:txt2]. While label propagation can be more computationally intensive, it often leads to improved accuracy by considering the global structure of the data.

Self-training has also been extended to more sophisticated architectures, such as the Noisy Student approach. In this method, a teacher model is trained on labeled data and then used to generate pseudo-labels for unlabeled data. A student model is then trained on both the original labeled data and the pseudo-labeled data, with added noise to the input during training to improve robustness. This process can be repeated iteratively, with the student model becoming the new teacher in subsequent rounds. This approach has demonstrated strong performance in image classification and other vision tasks [citation:txt3].

Despite its effectiveness, pseudo labeling is not without limitations. A major concern is confirmation bias, where the model becomes overly confident in its pseudo-labels and reinforces its own errors. To mitigate this, several strategies have been proposed. One such approach is MixUp, which blends pairs of data points and their corresponding labels, encouraging the model to produce smooth and consistent predictions across the input space [citation:txt17]. Another technique is to enforce a minimum number of labeled samples per mini-batch, ensuring that the model does not rely too heavily on pseudo-labels during training [citation:txt19]. This helps maintain a balance between the supervised and pseudo-supervised components of the loss function.

Meta Pseudo Labels (MPL) is another advanced method that addresses confirmation bias by continuously adapting the teacher model alongside the student. Instead of keeping the teacher fixed, as in vanilla pseudo labeling, MPL updates the teacher model in response to the student’s learning progress, allowing it to generate more accurate and adaptive pseudo-labels over time [citation:txt3]. This dynamic interaction between the teacher and student models has been shown to improve performance in SSL, particularly in scenarios where the pseudo-labels may initially be noisy or inaccurate.

In addition to reducing confirmation bias, other factors such as noise injection, data balancing, and the use of soft pseudo-labels have been shown to enhance the robustness of self-training. Soft pseudo-labels, which retain the full probability distribution rather than just the predicted class, allow for more nuanced training and can help the model better handle uncertainty [citation:txt20]. Techniques like FlexMatch and FreeMatch further refine pseudo labeling by introducing class-specific thresholds that adapt based on the learning progress of each class, ensuring that pseudo-labels are only used when they are most likely to be accurate [citation:txt19].

Overall, pseudo labeling and iterative self-training methods provide a powerful and flexible framework for semi-supervised learning when labeled data is limited. By combining these approaches with strategies to reduce confirmation bias and enhance pseudo-label quality, researchers have achieved significant improvements in model accuracy and generalization. These techniques are not only foundational but also serve as a basis for more advanced SSL methods, which will be explored in the following sections.

# Combining Pseudo Labeling with Consistency Regularization in Semi-Supervised Learning

Recent advances in semi-supervised learning have focused on combining pseudo labeling with consistency regularization to create more robust and data-efficient models. These methods integrate multiple strategies, including entropy minimization, MixUp, distribution alignment, and augmentation anchoring, to improve performance when labeled data is scarce. In this section, we examine key algorithms—MixMatch, ReMixMatch, DivideMix, and FixMatch—that exemplify this holistic approach.

## MixMatch: A Holistic Approach

MixMatch is a semi-supervised learning algorithm that combines consistency regularization, entropy minimization, and data augmentation techniques like MixUp to improve model accuracy with limited labeled data. The algorithm processes labeled and unlabeled data in parallel, generating augmented versions of both. For unlabeled data, MixMatch uses entropy minimization to enforce confidence in the model’s predictions and applies MixUp to blend data samples and their pseudo labels. This combination helps the model learn more robust representations and generalizes better to unseen data [citation:txt18].

The core mechanism of MixMatch involves applying weak and strong augmentations to unlabeled data. The weakly augmented version is used to generate a pseudo label, which is then combined with the strongly augmented version via MixUp. The model is trained to minimize the loss on both labeled and pseudo-labeled data, leading to significant performance improvements in low-label regimes [citation:txt11].

## ReMixMatch: Distribution Alignment and Augmentation Anchoring

ReMixMatch builds upon MixMatch by introducing two novel techniques: distribution alignment and augmentation anchoring. Distribution alignment ensures that the marginal distribution of predictions on unlabeled data is close to the marginal distribution of ground-truth labels. This helps mitigate the risk of the model learning incorrect distributions from noisy or biased unlabeled data [citation:txt12].

Augmentation anchoring, on the other hand, leverages the predictions of a weakly augmented version of an image as targets for its strongly augmented counterpart. This ensures that the model’s outputs remain consistent across different augmentation strategies, further enhancing robustness [citation:txt1]. ReMixMatch also incorporates a learned augmentation policy using AutoAugment, which is optimized during training to generate more effective augmentations [citation:txt13].

Empirically, ReMixMatch has demonstrated significant improvements in data efficiency. For example, on the CIFAR-10 dataset with only 250 labeled examples, ReMixMatch achieves 93.73% accuracy, outperforming MixMatch, which requires 4,000 labeled examples to reach similar performance [citation:txt15].

## DivideMix: Handling Noisy Labels with Co-Divide and Co-Guessing

DivideMix is designed to address the challenge of learning from noisy labels by integrating semi-supervised learning techniques. The algorithm trains two networks in parallel and uses a Gaussian Mixture Model (GMM) to dynamically divide the training data into a clean labeled set and a noisy unlabeled set. This co-divide strategy prevents confirmation bias by ensuring that each network uses the dataset division from the other [citation:txt16].

During the semi-supervised training phase, DivideMix enhances the MixMatch strategy by performing label co-refinement and co-guessing. For labeled samples, the ground-truth labels are refined using the predictions of the other network, while for unlabeled samples, the ensemble of both networks is used to make reliable label guesses [citation:txt7]. These strategies enable DivideMix to achieve state-of-the-art results on benchmark datasets with varying levels of label noise [citation:txt10].

## FixMatch: High-Confidence Pseudo Labeling with Weak and Strong Augmentation

FixMatch is a simplified yet effective semi-supervised learning algorithm that combines pseudo labeling with consistency regularization. The method generates pseudo labels for unlabeled data using a weakly augmented version of the input and filters out predictions with low confidence. Only high-confidence pseudo labels are used to train the model on strongly augmented data [citation:txt2].

The core idea behind FixMatch is to ensure that the model learns from reliable pseudo labels, reducing the risk of error propagation. By using a high confidence threshold (e.g., 0.95), FixMatch filters out noisy predictions and focuses on the most trustworthy pseudo labels [citation:txt4]. This approach has been shown to achieve performance close to fully supervised models, particularly in low-label regimes [citation:txt14].

FixMatch’s architecture is illustrated in the following diagram:
![](citation:img3)

As shown, the weakly and strongly augmented versions of the same image are processed by the model, and the predictions are compared using a consistency loss. This mechanism ensures that the model remains robust to input perturbations and learns from high-quality pseudo labels [citation:txt3].

In conclusion, these advanced methods—MixMatch, ReMixMatch, DivideMix, and FixMatch—demonstrate the effectiveness of combining pseudo labeling with consistency regularization. By integrating multiple strategies, they achieve significant improvements in model accuracy and robustness, even when labeled data is limited. These approaches form the foundation for further research into semi-supervised learning and its applications in real-world scenarios.

# Integrating Semi-Supervised Learning with Pre-Training

Recent advances in semi-supervised learning have demonstrated the effectiveness of combining self-supervised pre-training with task-specific fine-tuning and distillation. This integration leverages the strengths of both task-agnostic and task-specific learning, enabling models to achieve high performance even with limited labeled data. The framework typically involves three key steps: self-supervised pre-training on unlabeled data, supervised fine-tuning on a small labeled dataset, and distillation using unlabeled data to refine the model further. This approach has shown significant improvements in label efficiency, especially when using large neural networks [citation:txt3].

## Comparative Analysis of Pre-Training and Self-Training

Empirical studies have compared the performance of pre-training, self-training, and joint-training in different data regimes. One notable finding is that pre-training on unlabeled data, particularly with large models, is highly effective when labeled data is scarce. For instance, in the ImageNet classification task, using a large ResNet model pre-trained with SimCLRv2 and then fine-tuned on just 1% of the labeled data achieved 73.9% top-1 accuracy, a tenfold improvement in label efficiency over previous methods [citation:txt8]. In contrast, self-training methods, which generate pseudo-labels for unlabeled data and retrain the model iteratively, tend to be more effective in scenarios where the initial model is already reasonably accurate. However, self-training can suffer from confirmation bias if the pseudo-labels are noisy or incorrect [citation:txt12].

Joint-training approaches, which combine pre-training and self-training, often achieve the best performance by leveraging both the general representations learned from unlabeled data and the task-specific knowledge from labeled data. For example, the AS3L framework, which integrates active learning with pseudo-labeling, has demonstrated improvements of up to 5.4% over baseline methods in image classification tasks [citation:txt6]. These findings highlight the importance of selecting the right combination of methods based on the data regime and model architecture.

## Three-Step Procedure for Semi-Supervised Learning

A widely adopted three-step procedure for semi-supervised learning includes:

1. **Self-Supervised Pre-Training:** In this stage, a large neural network is trained on unlabeled data using self-supervised learning techniques such as contrastive learning. This step aims to learn general representations that can be adapted for downstream tasks [citation:txt4].

2. **Supervised Fine-Tuning:** After pre-training, the model is fine-tuned on a small labeled dataset to adapt the general representations to the specific task. This step is crucial for aligning the model with the task-specific objectives [citation:txt3].

3. **Distillation with Unlabeled Data:** Finally, the model is further refined by using unlabeled data to generate pseudo-labels and train a smaller student model. This step leverages the knowledge from the pre-trained and fine-tuned teacher model to improve the performance of the student model without the need for additional labeled data [citation:txt14].

This procedure is illustrated in Figure 3, where unlabeled data is used in two distinct ways: first for task-agnostic pre-training and second for task-specific distillation [citation:txt1]. The effectiveness of this approach is further supported by the fact that larger models are more label-efficient. For example, with 10% of the labeled data, a ResNet-50 model trained using this method achieved 77.5% top-1 accuracy, outperforming standard supervised training with the full labeled dataset [citation:txt8].

## Empirical Results on Model Size and Distillation

Empirical studies have consistently shown that model size plays a crucial role in the success of semi-supervised learning. Larger models are better at capturing the complexity of the data and are more robust to noisy pseudo-labels. Additionally, distillation using unlabeled data has been shown to improve performance by transferring knowledge from a larger teacher model to a smaller student model. This not only enhances the student model's accuracy but also makes it more compact and efficient [citation:txt11].

For instance, in the AS3L framework, distillation with unlabeled data significantly improved the performance of the model while also reducing the training time by about two-thirds [citation:txt6]. Similarly, in the ImageNet experiments, distillation helped refine the task-specific knowledge, leading to a substantial improvement in classification accuracy [citation:txt8]. These findings underscore the additive benefits of combining pre-training, fine-tuning, and distillation in semi-supervised learning.

![](citation:img1)

This structured approach to semi-supervised learning demonstrates the potential of integrating self-supervised pre-training with task-specific fine-tuning and distillation. By leveraging both labeled and unlabeled data effectively, these methods can significantly enhance model performance in scenarios where labeled data is limited.

# Best Practices and Practical Recommendations for Semi-Supervised Learning

Recent advances in semi-supervised learning have emphasized the importance of practical strategies to maximize performance while minimizing the risk of confirmation bias and model overfitting. A key takeaway from the literature is the critical role of data augmentation and label manipulation in improving generalization and robustness. Advanced data augmentation techniques, such as those used in ReMixMatch, encourage the model to learn from diverse transformations of the input data, thereby reducing overfitting and enhancing decision boundaries [citation:txt9]. For instance, strong augmentations combined with weak ones help ensure that the model produces consistent predictions across different versions of the same input [citation:txt15]. This consistency is further reinforced through methods like augmentation anchoring, which aligns predictions from weak and strong augmentations [citation:txt7].

MixUp, a technique that interpolates labeled and unlabeled data, has also emerged as a powerful tool in semi-supervised learning [citation:txt16]. However, it is not without its challenges. While MixUp can improve generalization in supervised settings, it may reduce the confidence of model predictions in semi-supervised scenarios, leading to noisy pseudo-labels [citation:txt8]. To mitigate this, researchers have proposed label sharpening, which reduces the entropy of the pseudo-label distribution, ensuring that the model retains high confidence in its predictions [citation:txt11]. This is typically achieved by adjusting the temperature parameter of the softmax function, effectively making the predicted distribution more deterministic [citation:txt18].

Another critical aspect of effective semi-supervised learning is the careful selection and filtering of pseudo-labels. Pseudo-labeling methods often rely on confidence thresholds to discard low-confidence predictions, as these are more likely to be incorrect and introduce noise [citation:txt12]. However, setting the threshold is a delicate balance—too low, and the model is exposed to noisy pseudo-labels; too high, and the model loses access to potentially useful data. Adaptive or class-conditional thresholds have been proposed to address this issue, allowing the model to leverage more data while still minimizing the risk of confirmation bias [citation:txt10].

Batch composition also plays a crucial role in training dynamics. As shown in several studies, maintaining a balanced ratio of labeled to unlabeled examples within each mini-batch is essential for stable and efficient training. For example, using 1 labeled example for every 99 unlabeled examples has been found to accelerate training and improve model performance when labeled data is limited [citation:txt1]. This strategy ensures that the model is constantly exposed to labeled data while benefiting from the large volume of unlabeled samples.

Finally, the integration of consistency regularization and pseudo-labeling with pre-training has demonstrated significant improvements in performance across a range of tasks [citation:txt4]. Techniques like FixMatch and its variants emphasize high-confidence pseudo-labels and consistency between weakly and strongly augmented inputs, leading to more accurate and reliable predictions [citation:txt14]. Moreover, newer methods such as Semi-Supervised RegMixup (SRM) combine pseudo-labeling with consistency regularization in a way that effectively utilizes both high- and low-confidence samples [citation:txt8].

In summary, successful semi-supervised learning hinges on a combination of well-designed data augmentation, careful pseudo-label filtering, strategic batch composition, and techniques like MixUp and label sharpening. These strategies, when implemented thoughtfully, can significantly enhance model performance in low-data regimes while mitigating the risk of confirmation bias.