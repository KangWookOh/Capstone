package halil.todolist.domain.member.service;

import halil.todolist.domain.member.dto.SignUpDto;
import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.repository.MemberRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
class MemberServiceTest {

    @Autowired
    MemberRepository memberRepository;

    @Autowired
    MemberService memberService;

    @Transactional
    @Test
    void signUp() {
        // given
        Member member = Member.builder()
                .email("test@email.com")
                .password("1234").build();

        SignUpDto signUpDto = new SignUpDto();
        signUpDto.setEmail(member.getEmail());
        signUpDto.setPassword(member.getPassword());

        Long id = memberService.signUp(signUpDto);

        // when
        Long savedId = memberRepository.findByEmail(signUpDto.getEmail()).get().getId();

        // then
        assertThat(id).isEqualTo(savedId);
    }
}