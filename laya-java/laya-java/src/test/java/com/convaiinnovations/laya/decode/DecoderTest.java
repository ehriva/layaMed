package com.convaiinnovations.laya.decode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

final class DecoderTest {

    /**
     * An empty action block is refused, not averaged.
     *
     * <p>This test asserted {@code {0.5f, 0.5f}} until now, and that assertion had stopped
     * matching the code it guards: {@code Decoder.actionProbabilities} throws on a zero-width
     * block. The implementation is the half that is right. {@code 0.5} is a number no model
     * produced, published into the field callers gate escalation on, and the reference does not
     * produce it either — {@code laya/onnx_agent.py} indexes {@code act[offset + r, 0]}, so zero
     * columns raise there. A port that answers where the reference raises is not a port.
     *
     * <p>The message is asserted, not just the type, because that is what a caller reads.
     */
    @Test
    @DisplayName("an empty action block is refused, not averaged")
    void emptyActionLogitsAreRefused() {
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> Decoder.actionProbabilities(new float[0]));
        assertEquals("the graph produced no action logits, so there is no action probability",
                refused.getMessage());
    }
}
