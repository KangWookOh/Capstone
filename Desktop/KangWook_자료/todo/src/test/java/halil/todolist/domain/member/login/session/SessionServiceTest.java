package halil.todolist.domain.member.login.session;

import halil.todolist.domain.member.dto.SignUpDto;
import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.service.MemberService;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.transaction.annotation.Transactional;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
class SessionServiceTest {

    @Autowired
    MemberService memberService;

    @Autowired
    SessionService sessionservice;

    @DisplayName("세션 생성, 조회")
    @Transactional
    @Test
    void createSession() {
        // given
        MockHttpServletResponse response = new MockHttpServletResponse();
        MockHttpServletRequest request = new MockHttpServletRequest();

        Member member = Member.builder()
                .email("test@email.com")
                .password("1234").build();

        SignUpDto signUpDto = new SignUpDto();
        signUpDto.setEmail(member.getEmail());
        signUpDto.setPassword(member.getPassword());

        sessionservice.createSession(signUpDto, response);
        request.setCookies(response.getCookies());

        // when
        Object session = sessionservice.getSession(request);

        // then
        assertThat(session).isEqualTo(signUpDto);
    }

    @DisplayName("세션 만료")
    @Transactional
    @Test
    void expiredSession() {
        // given
        MockHttpServletResponse response = new MockHttpServletResponse();
        MockHttpServletRequest request = new MockHttpServletRequest();

        Member member = Member.builder()
                .email("test@email.com")
                .password("1234").build();

        SignUpDto signUpDto = new SignUpDto();
        signUpDto.setEmail(member.getEmail());
        signUpDto.setPassword(member.getPassword());

        sessionservice.createSession(signUpDto, response);
        request.setCookies(response.getCookies());

        // when
        sessionservice.expire(request);
        Object session = sessionservice.getSession(request);

        // then
        assertThat(session).isNull();
    }
}